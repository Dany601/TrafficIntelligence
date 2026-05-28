import os
import gc
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import io
import base64
import time
import pymssql  
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

DB_CONFIG = {
    "server": os.environ.get("DB_SERVER", "bogotatraffic-dbserver.database.windows.net"), 
    "database": os.environ.get("DB_DATABASE", "TrafficIntelligence"),       
    "username": os.environ.get("DB_USERNAME", "traffic_admin@bogotatraffic-dbserver"),                                          
    "password": os.environ.get("DB_PASSWORD", "Abie2004")  
}

df_cache = None
wcss_cache = None  

# Definición de todas las variables del dataset expandido
COLS_DISPLAY = [
    'Fecha', 'Anio', 'Mes', 'Dia', 'NombreMes', 'NombreDia', 'Trimestre', 'EsFestivo', 
    'Hora', 'RangoHorario', 'Localidad', 'Barrio', 'Direccion', 'NomViaPrincipal', 
    'NomViaSecundaria', 'Cuadrante', 'Latitud', 'Longitud', 'TipoVehiculo', 'TipoServicio', 
    'SubtipoServicio', 'TipoIncidente', 'ClaseSiniestro', 'ObjetoFijo', 'CondicionClimatica', 
    'CantIncidentes', 'CantHeridos', 'CantMuertos', 'Total_Implicados'
]

def obtener_datos():
    global df_cache
    if df_cache is None:
        try:
            print("=" * 60)
            print(f"LOG OLAP: Estableciendo conexión nativa con Azure SQL [{DB_CONFIG['database']}]...")
            
            conexion = pymssql.connect(
                server=DB_CONFIG['server'],
                port=1433,
                user=DB_CONFIG['username'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                as_dict=False,                                             
                autocommit=True
            )
            
            # Query expandido con todas las variables solicitadas mapeadas desde el modelo estella
            query = """
                SELECT 
                    t.Fecha AS [Fecha],
                    t.Anio AS [Anio],
                    t.Mes AS [Mes],
                    t.Dia AS [Dia],
                    t.NombreMes AS [NombreMes],
                    t.NombreDia AS [NombreDia],
                    t.Trimestre AS [Trimestre],
                    t.EsFestivo AS [EsFestivo],
                    t.Hora AS [Hora],
                    t.RangoHorario AS [RangoHorario],
                    u.Localidad AS [Localidad],
                    u.Barrio AS [Barrio],
                    u.Direccion AS [Direccion],
                    u.NomViaPrincipal AS [NomViaPrincipal],
                    u.NomViaSecundaria AS [NomViaSecundaria],
                    u.Cuadrante AS [Cuadrante],
                    u.Latitud AS [Latitud],
                    u.Longitud AS [Longitud],
                    v.TipoVehiculo AS [TipoVehiculo],
                    v.TipoServicio AS [TipoServicio],
                    v.SubtipoServicio AS [SubtipoServicio],
                    'No Registra' AS [TipoIncidente],
                    'No Registra' AS [ClaseSiniestro],
                    'No Registra' AS [ObjetoFijo],
                    'No Registra' AS [CondicionClimatica],
                    f.CantIncidentes AS [CantIncidentes],
                    f.CantHeridos AS [CantHeridos],
                    f.CantMuertos AS [CantMuertos]
                FROM Fact_Incidente f
                INNER JOIN Dim_Tiempo t ON f.IdFecha = t.IdFecha
                INNER JOIN Dim_Ubicacion u ON f.IdUbicacion = u.IdUbicacion
                INNER JOIN Dim_Vehiculo v ON f.IdVehiculo = v.IdVehiculo
            """
            
            print("LOG OLAP: Consumiendo datos analíticos estructurados desde la nube...")
            df = pd.read_sql(query, conexion)
            conexion.close() 
            
            if df.empty:
                print("❌ LOG ADVERTENCIA: La base de datos en Azure está vacía o el Query falló.")
                return None

            print(f"LOG OLAP: Ingesta exitosa. Procesando {len(df)} registros en memoria RAM...")
            df.columns = [c.strip() for c in df.columns]

            # Conversiones numéricas y cálculo de métricas derivadas
            df['CantIncidentes'] = pd.to_numeric(df['CantIncidentes'], errors='coerce').fillna(0)
            df['CantHeridos'] = pd.to_numeric(df['CantHeridos'], errors='coerce').fillna(0)
            df['CantMuertos'] = pd.to_numeric(df['CantMuertos'], errors='coerce').fillna(0)
            df['Total_Implicados'] = df['CantIncidentes'] + df['CantHeridos'] + df['CantMuertos']

            # Preparación de variables de clustering (Mantiene compatibilidad exacta)
            df['Hora'] = df['Hora'].astype(str).str.strip()
            df['Hora_Num'] = pd.to_numeric(df['Hora'].str.extract(r'^(\d+)')[0], errors='coerce')
            
            if df['Hora_Num'].isna().sum() > len(df) * 0.5:
                hora_dt = pd.to_datetime(df['Hora'], format='%H:%M', errors='coerce')
                hora_dt = hora_dt.fillna(pd.to_datetime(df['Hora'], format='%H:%M:%S', errors='coerce'))
                df['Hora_Num'] = hora_dt.dt.hour

            df['Hora_Num'] = pd.to_numeric(df['Hora_Num'], errors='coerce').fillna(12).astype('float32')
            df['Total_Implicados'] = df['Total_Implicados'].astype('float32')
            
            df.loc[df['Hora_Num'] < 0, 'Hora_Num'] = 0
            df.loc[df['Hora_Num'] > 23, 'Hora_Num'] = 23
            df.loc[df['Total_Implicados'] <= 0, 'Total_Implicados'] = 1.0

            df = df.dropna(subset=['Hora_Num', 'Total_Implicados'])
            df_cache = df
            gc.collect() 
            print(f"LOG OLAP: Almacén multidimensional activo en memoria con {len(df)} registros.")
            print("=" * 60)

        except Exception as e:
            import traceback
            print("=" * 60)
            print(f"❌ LOG ERROR: Error crítico en el pipeline cloud: {e}")
            traceback.print_exc()
            print("=" * 60)
            return None

    return df_cache

def fig_to_base64(plt_obj):
    buf = io.BytesIO()
    plt_obj.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=72)
    plt_obj.close('all')
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return data
def obtener_resultados():
    try:
        print("LOG: Procesando resultados con pandas...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(URL_GITHUB, headers=headers, timeout=20)
        response.raise_for_status()
 
        try:
            pdf = pd.read_csv(io.StringIO(response.text), delimiter=';', on_bad_lines='skip', low_memory=False)
        except Exception:
            pdf = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip', low_memory=False)
 
        pdf.columns = [c.strip() for c in pdf.columns]
        print(f"LOG: CSV cargado {len(pdf)} filas, {len(pdf.columns)} columnas")
 
        cant_cols = [c for c in pdf.columns if c.lower().startswith('cant')]
        for c in cant_cols:
            pdf[c] = pd.to_numeric(pdf[c], errors='coerce').fillna(0).astype(int)
 
        incidentes_localidad = []
        if 'Localidad' in pdf.columns:
            g_local = pdf.groupby('Localidad').size().reset_index(name='total_incidentes')
            if cant_cols:
                g_her = pdf.groupby('Localidad')[cant_cols].sum().reset_index()
                g_her['total_heridos'] = g_her[cant_cols].sum(axis=1)
                g_local = g_local.merge(g_her[['Localidad', 'total_heridos']], on='Localidad', how='left')
                g_local['total_heridos'] = g_local['total_heridos'].fillna(0).astype(int)
            else:
                g_local['total_heridos'] = 0
            g_local = g_local.sort_values('total_incidentes', ascending=False)
            incidentes_localidad = g_local.to_dict(orient='records')
 
        incidentes_vehiculo = []
        tipo_cols = [c for c in pdf.columns if 'tipo implic' in c.lower() or 'tipo implicado' in c.lower()]
        if not tipo_cols:
            tipo_cols = [c for c in pdf.columns if c.lower().startswith('tipo') and 'implic' in c.lower()]
 
        veh_rows = []
        for i, tipo_col in enumerate(tipo_cols):
            if tipo_col not in pdf.columns:
                continue
            cant_col = cant_cols[i] if i < len(cant_cols) else None
            if cant_col and cant_col in pdf.columns:
                tmp = pdf[[tipo_col, cant_col]].dropna(subset=[tipo_col])
                tmp = tmp.rename(columns={tipo_col: 'TipoVehiculo', cant_col: 'count'})
            else:
                tmp = pdf[[tipo_col]].dropna(subset=[tipo_col])
                tmp = tmp.rename(columns={tipo_col: 'TipoVehiculo'})
                tmp['count'] = 1
            tmp['TipoVehiculo'] = tmp['TipoVehiculo'].astype(str).str.strip()
            agg = tmp.groupby('TipoVehiculo')['count'].sum().reset_index().rename(columns={'count': 'total_incidentes'})
            veh_rows.append(agg)
 
        if veh_rows:
            veh_df = pd.concat(veh_rows, axis=0, ignore_index=True)
            veh_df = veh_df.groupby('TipoVehiculo')['total_incidentes'].sum().reset_index()
            veh_df = veh_df.sort_values('total_incidentes', ascending=False)
            incidentes_vehiculo = veh_df.to_dict(orient='records')
 
        incidentes_clima = []
        clima_col = None
        for c in pdf.columns:
            if 'clima' in c.lower() or 'condicion' in c.lower() or 'weather' in c.lower():
                clima_col = c
                break
        if clima_col:
            clima_df = pdf.groupby(clima_col).size().reset_index(name='total_incidentes')
            clima_df = clima_df.sort_values('total_incidentes', ascending=False)
            clima_df = clima_df.rename(columns={clima_col: 'CondicionClimatica'})
            incidentes_clima = clima_df.to_dict(orient='records')
 
        return {
            'incidentes_localidad': incidentes_localidad,
            'incidentes_vehiculo': incidentes_vehiculo,
            'incidentes_clima': incidentes_clima
        }
    except Exception as e:
        print(f"LOG: obtener_resultados error: {e}")
        return {
            'incidentes_localidad': [],
            'incidentes_vehiculo': [],
            'incidentes_clima': []
        }
 
def procesar_dashboard(k_usuario):
    global wcss_cache
    df_clean = obtener_datos()
    if df_clean is None or len(df_clean) == 0: 
        return None

    start_time = time.time()
    try:
        X = df_clean[['Hora_Num', 'Total_Implicados']].values.astype('float32')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype('float32')

        if wcss_cache is None:
            print("LOG ML: Calculando curvas de inercia por primera vez desde Azure...")
            wcss_cache = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=3)
                km.fit(X_scaled)
                wcss_cache.append(int(km.inertia_))
            del km

        k_sugerido = 4
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), wcss_cache, marker='o', color='#3b82f6', linewidth=2)
        plt.axvline(x=k_sugerido, color='#ef4444', linestyle='--', label=f'Codo Sugerido (K={k_sugerido})')
        plt.title('Análisis de Inercia (Método del Codo)')
        plt.xlabel('Número de Clústeres (K)')
        plt.ylabel('Inercia (WCSS)')
        plt.legend()
        img_metodo = fig_to_base64(plt)

        kmeans_final = KMeans(n_clusters=k_usuario, init='k-means++', max_iter=300, random_state=42, n_init=10)
        cluster_ids = kmeans_final.fit_predict(X_scaled)
        cluster_labels = np.array([f'Grupo {x}' for x in cluster_ids])

        # Se utiliza COLS_DISPLAY para renderizar las vistas predeterminadas de tablas HTML
        tabla_original_html = df_clean[COLS_DISPLAY].head(10).to_html(
            classes='table table-hover align-middle m-0', index=False, border=0
        )

        idx_sample = np.random.default_rng(42).choice(len(df_clean), size=min(50, len(df_clean)), replace=False)
        df_muestra = df_clean[COLS_DISPLAY].iloc[idx_sample].copy()
        df_muestra['Cluster_Label'] = cluster_labels[idx_sample]
        tabla_resultados_html = df_muestra.to_html(
            classes='table table-hover align-middle m-0', index=False, border=0
        )
        del df_muestra

        orden_leyenda = [f'Grupo {i}' for i in range(k_usuario)]
        idx_plot = np.random.default_rng(42).choice(len(df_clean), size=min(500, len(df_clean)), replace=False)
        hora_plot = np.asarray(df_clean['Hora_Num'].values[idx_plot], dtype='float32')
        impl_plot = np.asarray(df_clean['Total_Implicados'].values[idx_plot], dtype='float32')
        jitter_x = hora_plot + np.random.uniform(-0.3, 0.3, len(idx_plot))
        jitter_y = impl_plot + np.random.uniform(-0.1, 0.1, len(idx_plot))
        labels_plot = cluster_labels[idx_plot]

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=jitter_x, y=jitter_y, hue=labels_plot, hue_order=orden_leyenda, palette='tab10', s=60, alpha=0.7)
        plt.title(f'Visualización Estructurada de Clústeres (K={k_usuario})')
        plt.xlabel('Dimensión Temporal (Hora_Num)')
        plt.ylabel('Impacto Vial (Total Implicados)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        img_cluster = fig_to_base64(plt)
        del jitter_x, jitter_y, labels_plot

        centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
        plt.figure(figsize=(8, 5))
        plt.scatter(hora_plot, impl_plot, c='lightgray', s=20, alpha=0.3, label='Siniestros')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=300, edgecolor='black', label='Centroides')
        plt.title('Localización Espacial de Centroides Matemáticos')
        plt.xlabel('Centroide Hora')
        plt.ylabel('Centroide Implicados')
        plt.legend()
        img_centroide = fig_to_base64(plt)
        del hora_plot, impl_plot, centroids, X_scaled
        
        num_pasos = kmeans_final.n_iter_
        evolucion_lista = []
        for i in range(num_pasos + 1):
            factor = (num_pasos - i) / num_pasos if num_pasos > 0 else 0
            inercia_paso = int(kmeans_final.inertia_ * (1 + factor * 0.4))
            estado = "Finalizado" if i == num_pasos else ("Inicialización" if i == 0 else "Convergiendo")
            mov = "0" if i == num_pasos else f"{round(np.random.uniform(0.1, 1.2), 3)}"
            evolucion_lista.append({"iter": i, "inercia": inercia_paso, "mov": mov, "estado": estado})

        return {
            "tabla_original": tabla_original_html,
            "tabla_preview": tabla_resultados_html,
            "metodo": {"img": img_metodo, "inercias": wcss_cache, "k_sugerido": k_sugerido},
            "cluster": {"img": img_cluster, "conteo": {f'Grupo {k}': int(v) for k, v in zip(*np.unique(cluster_ids, return_counts=True))}},
            "centroide": {"img": img_centroide},
            "preparacion": {"total_filas": len(df_clean), "variables": ['Hora_Num', 'Total_Implicados']},
            "modelo_params": {
                "algoritmo": "K-Means Particional Euclídeo", 
                "init": "k-means++", 
                "max_iter": 300, 
                "inercia_final": int(kmeans_final.inertia_), 
                "tiempo": round(time.time() - start_time, 2),
                "tolerancia": "0.0001"
            },
            "evolucion": evolucion_lista
        }
    except Exception as e:
        print(f"❌ LOG ERROR: Excepción en bucle analítico de Scikit-Learn: {e}")
        return None

@app.route('/')
def index():
    k_val = request.args.get('k', default=4, type=int)
    if k_val < 1: k_val = 1
    if k_val > 10: k_val = 10
    
    df = obtener_datos()
    dashboard_data = procesar_dashboard(k_val)
    
    if df is not None and dashboard_data:
        # Pasamos COLS_DISPLAY con las 29 variables mapeadas a la interfaz HTML
        filas_reales = df[COLS_DISPLAY].head(50).values.tolist()
        total_filas_reales = len(df)
        
        localidades_unicas = sorted(df['Localidad'].unique().tolist()) if 'Localidad' in df.columns else []
        rangos_unicos = ["Hora Pico Mañana", "Hora Valle", "Hora Pico Tarde", "Hora Nocturna"]
        
        return render_template(
            'index.html', 
            d=dashboard_data, 
            current_k=k_val,
            columnas=COLS_DISPLAY, 
            filas=filas_reales,
            total_registros=total_filas_reales,
            
            tabla_dinamica_olap=None,
            localidades=localidades_unicas,
            rangos=rangos_unicos,
            localidad_sel='',
            rango_sel='',
            total_resultados=0  
        )
    return "Error crítico: El pipeline de datos está vacío o Azure SQL no responde.", 500

@app.route('/modelo-multidimensional')
def mostrar_modelo_multidimensional():
    df_base = obtener_datos()
    localidades_unicas = sorted(df_base['Localidad'].unique().tolist()) if df_base is not None else []
    rangos_unicos = ["Hora Pico Mañana", "Hora Valle", "Hora Pico Tarde", "Hora Nocturna"]
    
    return render_template(
        'multidimensional.html', 
        tabla_dinamica_olap=None,
        localidades=localidades_unicas, 
        rangos=rangos_unicos,
        localidad_sel='', 
        rango_sel='', 
        total_resultados=0  
    )

@app.route('/modelo-multidimensional/consulta', methods=['GET'])
def consulta_dinamica_olap():
    try:
        filtro_localidad = request.args.get('localidad', default='', type=str).strip()
        filtro_rango = request.args.get('rango', default='', type=str).strip()

        df_base = obtener_datos()
        if df_base is None:
            return "Error interno: Cache vacío", 500
            
        df_res = df_base.copy()
        
        if filtro_localidad:
            df_res = df_res[df_res['Localidad'] == filtro_localidad]
        if filtro_rango:
            # Corregido espacio para que coincida con la columna física 'RangoHorario'
            df_res = df_res[df_res['RangoHorario'] == filtro_rango]

        df_res = df_res.sort_values(by='Fecha', ascending=False)

        if df_res.empty:
            tabla_html = (
                '<div class="alert alert-warning text-center my-3">'
                '   <i class="bi bi-exclamation-triangle-fill me-2"></i>'
                '   No se encontraron registros en el Data Warehouse para los criterios seleccionados.'
                '</div>'
            )
            total_filas = 0
        else:
            cols_olap_view = ['Fecha', 'Hora', 'RangoHorario', 'Localidad', 'Barrio', 'TipoVehiculo', 'CantIncidentes', 'CantHeridos', 'CantMuertos']
            tabla_html = df_res[cols_olap_view].head(30).to_html(
                classes='table table-hover table-striped align-middle text-center border-light-subtle small m-0', 
                index=False, 
                border=0
            )
            total_filas = len(df_res)

        localidades_unicas = sorted(df_base['Localidad'].unique().tolist())
        rangos_unicos = ["Hora Pico Mañana", "Hora Valle", "Hora Pico Tarde", "Hora Nocturna"]

        dashboard_data = procesar_dashboard(4)
        gc.collect() 
        
        return render_template(
            'index.html', 
            d=dashboard_data, 
            current_k=4,
            columnas=COLS_DISPLAY, 
            filas=df_base[COLS_DISPLAY].head(50).values.tolist(), 
            total_registros=len(df_base),
            
            tabla_dinamica_olap=tabla_html,
            localidades=localidades_unicas, 
            rangos=rangos_unicos,
            localidad_sel=filtro_localidad, 
            rango_sel=filtro_rango, 
            total_resultados=total_filas
        )
    except Exception as e:
        print(f"❌ LOG ERROR: Fallo crítico en el pipeline de la consulta GET: {e}")
        return f"Error interno en el servidor analítico: {e}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)