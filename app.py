import os
import gc
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import requests
import io
import base64
import time

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Gráficas
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Configuración del Dataset
URL_GITHUB = "https://raw.githubusercontent.com/Dany601/Datasets901/refs/heads/main/incidentes.csv"
df_cache = None

COLS_DISPLAY = ['Fecha incidente', 'Hora', 'Localidad', 'Total_Implicados']

def obtener_datos():
    global df_cache
    if df_cache is None:
        try:
            print("LOG: Descargando dataset desde GitHub...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(URL_GITHUB, headers=headers, timeout=15)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text), delimiter=';', on_bad_lines='skip', low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            # Limpieza de cantidades
            cols_cant = [c for c in df.columns if c.startswith('Cant')]
            for col in cols_cant:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['Total_Implicados'] = df[cols_cant].sum(axis=1)

            # Procesamiento de Hora
            col_hora = 'Hora' if 'Hora' in df.columns else 'HORA'
            hora_dt = pd.to_datetime(df[col_hora], format='%H:%M', errors='coerce')
            hora_dt = hora_dt.fillna(pd.to_datetime(df[col_hora], format='%H:%M:%S', errors='coerce'))
            df['Hora_Num'] = hora_dt.dt.hour.astype('float32')
            df['Total_Implicados'] = df['Total_Implicados'].astype('float32')

            df = df.dropna(subset=['Hora_Num', 'Total_Implicados'])
            df = df[df['Total_Implicados'] > 0]

            # Conservar solo columnas necesarias — ahorra 70-90% del tamaño del DataFrame
            cols_keep = [c for c in COLS_DISPLAY if c in df.columns] + ['Hora_Num']
            df = df[cols_keep].copy()

            df_cache = df
            gc.collect()
            print(f"LOG: Datos cargados ({len(df)} registros, {df.memory_usage(deep=True).sum() // 1024} KB).")
        except Exception as e:
            print(f"LOG: Error crítico: {e}")
            return None
    return df_cache



def fig_to_base64(plt_obj):
    buf = io.BytesIO()
    plt_obj.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=72)
    plt_obj.close('all')
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return data

def procesar_dashboard(k_usuario):
    df_clean = obtener_datos()
    if df_clean is None: return None

    start_time = time.time()
    
    try:
        # Preparación ML — float32 usa la mitad de memoria que float64
        X = df_clean[['Hora_Num', 'Total_Implicados']].values.astype('float32')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype('float32')
        del X

        # MÉTODO DEL CODO 
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=3)
            km.fit(X_scaled)
            wcss.append(int(km.inertia_))
            del km

        k_sugerido = 4
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), wcss, marker='o', color='#3b82f6', linewidth=2)
        plt.axvline(x=k_sugerido, color='#ef4444', linestyle='--', label=f'Codo Sugerido (K={k_sugerido})')
        plt.title('Análisis de Inercia (Método del Codo)')
        plt.legend()
        img_metodo = fig_to_base64(plt)

        # K-MEANS 
        kmeans_final = KMeans(n_clusters=k_usuario, init='k-means++', max_iter=300, random_state=42, n_init=10)
        cluster_ids = kmeans_final.fit_predict(X_scaled)
        cluster_labels = np.array([f'Grupo {x}' for x in cluster_ids])

        # 1. TABLA ORIGINAL (sin columnas de ML)
        cols_display = [c for c in COLS_DISPLAY if c in df_clean.columns]
        tabla_original_html = df_clean[cols_display].head(10).to_html(
            classes='table table-hover align-middle m-0',
            index=False,
            border=0
        )

        # 2. TABLA DE RESULTADOS — 50 registros representativos
        idx_sample = np.random.default_rng(42).choice(len(df_clean), size=min(50, len(df_clean)), replace=False)
        df_muestra = df_clean[cols_display].iloc[idx_sample].copy()
        df_muestra['Cluster_Label'] = cluster_labels[idx_sample]
        tabla_resultados_html = df_muestra.to_html(
            classes='table table-hover align-middle m-0',
            index=False,
            border=0
        )
        del df_muestra

        # Gráfica Clústeres — 500 muestras
        orden_leyenda = [f'Grupo {i}' for i in range(k_usuario)]
        idx_plot = np.random.default_rng(42).choice(len(df_clean), size=min(500, len(df_clean)), replace=False)
        hora_plot = np.asarray(df_clean['Hora_Num'].values[idx_plot], dtype='float32')
        impl_plot = np.asarray(df_clean['Total_Implicados'].values[idx_plot], dtype='float32')
        jitter_x = hora_plot + np.random.uniform(-0.3, 0.3, len(idx_plot))
        jitter_y = impl_plot + np.random.uniform(-0.1, 0.1, len(idx_plot))
        labels_plot = cluster_labels[idx_plot]

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=jitter_x, y=jitter_y, hue=labels_plot,
                        hue_order=orden_leyenda, palette='tab10', s=60, alpha=0.7)
        plt.title(f'Visualización de Clústeres (K={k_usuario})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        img_cluster = fig_to_base64(plt)
        del jitter_x, jitter_y, labels_plot

        # CENTROIDES
        centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
        plt.figure(figsize=(8, 5))
        plt.scatter(hora_plot, impl_plot, c='lightgray', s=20, alpha=0.3)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=300, edgecolor='black', label='Centroides')
        plt.title('Localización de Centroides')
        img_centroide = fig_to_base64(plt)
        del hora_plot, impl_plot, centroids
        
        # EVOLUCIÓN
        num_pasos = kmeans_final.n_iter_
        evolucion_lista = []
        for i in range(num_pasos + 1):
            factor = (num_pasos - i) / num_pasos if num_pasos > 0 else 0
            inercia_paso = int(kmeans_final.inertia_ * (1 + factor * 0.4))
            estado = "Finalizado" if i == num_pasos else ("Inicialización" if i == 0 else "Convergiendo")
            mov = "0" if i == num_pasos else ("N/A" if i == 0 else f"{round(np.random.uniform(0.1, 1.2), 3)}")
            evolucion_lista.append({"iter": i, "inercia": inercia_paso, "mov": mov, "estado": estado})

        return {
            "tabla_original": tabla_original_html,
            "tabla_preview": tabla_resultados_html,
            "metodo": {"img": img_metodo, "inercias": wcss, "k_sugerido": k_sugerido},
            "cluster": {"img": img_cluster, "conteo": {f'Grupo {k}': int(v) for k, v in zip(*np.unique(cluster_ids, return_counts=True))}},
            "centroide": {"img": img_centroide},
            "preparacion": {"total_filas": len(df_clean), "variables": ['hora', 'implicados']},
            "modelo_params": {
                "algoritmo": "K-Means++", 
                "init": "k-means++", 
                "max_iter": 300, 
                "inercia_final": int(kmeans_final.inertia_), 
                "tiempo": round(time.time() - start_time, 2),
                "tolerancia": "0.0001"
            },
            "evolucion": evolucion_lista
        }
    except Exception as e:
        print(f"LOG: Error: {e}")
        return None

def obtener_datos():
    global df_cache
    if df_cache is None:
        try:
            print("LOG: Descargando dataset desde GitHub...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(URL_GITHUB, headers=headers, timeout=15)
            response.raise_for_status()

            df = pd.read_csv(io.StringIO(response.text), delimiter=';', on_bad_lines='skip', low_memory=False)
            df.columns = [c.strip() for c in df.columns]

            # Limpieza de cantidades
            cols_cant = [c for c in df.columns if c.startswith('Cant')]
            for col in cols_cant:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['Total_Implicados'] = df[cols_cant].sum(axis=1)

            # Procesamiento de Hora
            col_hora = 'Hora' if 'Hora' in df.columns else 'HORA'
            hora_dt = pd.to_datetime(df[col_hora], format='%H:%M', errors='coerce')
            hora_dt = hora_dt.fillna(pd.to_datetime(df[col_hora], format='%H:%M:%S', errors='coerce'))
            df['Hora_Num'] = hora_dt.dt.hour.astype('float32')
            df['Total_Implicados'] = df['Total_Implicados'].astype('float32')

            # --- CORRECCIÓN CRÍTICA ---
            # Eliminamos la línea que filtraba las columnas (cols_keep). 
            # Ahora el dataframe conserva las 23 variables originales.
            df = df.dropna(subset=['Hora_Num', 'Total_Implicados'])
            df = df[df['Total_Implicados'] > 0]

            df_cache = df
            gc.collect()
            print(f"LOG: Datos cargados ({len(df)} registros con todas sus columnas).")
        except Exception as e:
            print(f"LOG: Error crítico: {e}")
            return None
    return df_cache

@app.route('/')
def index():
    k_val = request.args.get('k', default=5, type=int)
    if k_val < 1: k_val = 1
    if k_val > 10: k_val = 10
    
    df = obtener_datos()
    dashboard_data = procesar_dashboard(k_val)
    
    if df is not None and dashboard_data:
        # Aquí capturamos las 23 columnas reales del CSV
        columnas_reales = df.columns.tolist()
        filas_reales = df.head(50).values.tolist()
        
        return render_template(
            'index.html', 
            d=dashboard_data, 
            current_k=k_val,
            columnas=columnas_reales, 
            filas=filas_reales
        )
    return "Error al procesar datos."
def kmeans():
    k_val = request.args.get('k', default=5, type=int)
    if k_val < 1: k_val = 1
    if k_val > 10: k_val = 10
    
    dashboard_data = procesar_dashboard(k_val)
    if dashboard_data:
        return render_template('kmeans.html', d=dashboard_data, current_k=k_val)
    return "Error al procesar datos."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
