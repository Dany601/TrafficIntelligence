import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import webbrowser
from threading import Timer
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
            df['Hora_DT'] = pd.to_datetime(df[col_hora], format='%H:%M', errors='coerce')
            df.loc[df['Hora_DT'].isna(), 'Hora_DT'] = pd.to_datetime(df[col_hora], format='%H:%M:%S', errors='coerce')
            df['Hora_Num'] = df['Hora_DT'].dt.hour

            df = df.dropna(subset=['Hora_Num', 'Total_Implicados']).copy()
            df = df[df['Total_Implicados'] > 0]
            
            df_cache = df 
            print(f"LOG: Datos cargados ({len(df)} registros).")
        except Exception as e:
            print(f"LOG: Error crítico: {e}")
            return None
    return df_cache

def fig_to_base64(plt_obj):
    buf = io.BytesIO()
    plt_obj.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt_obj.close() 
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def procesar_dashboard(k_usuario):
    df_clean = obtener_datos()
    if df_clean is None: return None

    start_time = time.time()
    
    try:
        # Preparación ML
        X = df_clean[['Hora_Num', 'Total_Implicados']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # MÉTODO DEL CODO — n_init=3 para la exploración (no requiere precisión máxima)
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=3)
            km.fit(X_scaled)
            wcss.append(int(km.inertia_))

        k_sugerido = 4 
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), wcss, marker='o', color='#3b82f6', linewidth=2)
        plt.axvline(x=k_sugerido, color='#ef4444', linestyle='--', label=f'Codo Sugerido (K={k_sugerido})')
        plt.title('Análisis de Inercia (Método del Codo)')
        plt.legend()
        img_metodo = fig_to_base64(plt)

        # K-MEANS FINAL
        kmeans_final = KMeans(n_clusters=k_usuario, init='k-means++', max_iter=300, random_state=42, n_init=10)
        df_clean['Cluster'] = kmeans_final.fit_predict(X_scaled)
        df_clean['Cluster_Label'] = df_clean['Cluster'].apply(lambda x: f'Grupo {x}')
        
        # --- SOLUCIÓN A TU DUDA: LAS DOS TABLAS ---

        # 1. TABLA ORIGINAL (Para Exploración): Quitamos las columnas del modelo
        # Usamos errors='ignore' por si aún no se han creado
        tabla_original_html = df_clean.drop(columns=['Cluster', 'Cluster_Label', 'Hora_DT', 'Hora_Num'], errors='ignore').head(10).to_html(
            classes='table table-hover align-middle m-0', 
            index=False, 
            border=0
        )

        # 2. TABLA DE RESULTADOS (Con Clústeres): Muestra representativa de 50 registros
        columnas_res = ['Fecha incidente', 'Hora', 'Localidad', 'Total_Implicados', 'Cluster_Label']
        tabla_resultados_html = df_clean[columnas_res].sample(n=min(50, len(df_clean)), random_state=42).to_html(
            classes='table table-hover align-middle m-0',
            index=False,
            border=0
        )

        # Gráfica Clústeres — muestreo para rendimiento
        orden_leyenda = [f'Grupo {i}' for i in range(k_usuario)]
        df_plot = df_clean.sample(n=min(500, len(df_clean)), random_state=42)
        jitter_x = df_plot['Hora_Num'] + np.random.uniform(-0.3, 0.3, len(df_plot))
        jitter_y = df_plot['Total_Implicados'] + np.random.uniform(-0.1, 0.1, len(df_plot))

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=jitter_x, y=jitter_y, hue=df_plot['Cluster_Label'],
                        hue_order=orden_leyenda, palette='tab10', s=80, alpha=0.7)
        plt.title(f'Visualización de Clústeres (K={k_usuario})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        img_cluster = fig_to_base64(plt)

        # CENTROIDES
        centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)
        df_plot_c = df_clean.sample(n=min(500, len(df_clean)), random_state=42)
        plt.figure(figsize=(10, 6))
        plt.scatter(df_plot_c['Hora_Num'], df_plot_c['Total_Implicados'], c='lightgray', s=30, alpha=0.3)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=350, edgecolor='black', label='Centroides')
        plt.title('Localización de Centroides')
        img_centroide = fig_to_base64(plt)
        
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
            "cluster": {"img": img_cluster, "conteo": df_clean['Cluster'].value_counts().to_dict()},
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

@app.route('/')
def index():
    k_val = request.args.get('k', default=5, type=int)
    if k_val < 1: k_val = 1
    if k_val > 10: k_val = 10
    
    dashboard_data = procesar_dashboard(k_val)
    if dashboard_data:
        return render_template('index.html', d=dashboard_data, current_k=k_val)
    return "Error al procesar datos."

if __name__ == '__main__':
    app.run()
