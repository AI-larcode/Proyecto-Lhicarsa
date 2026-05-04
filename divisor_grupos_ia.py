import os, re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ================================
# 1. CONFIGURACIÓN
# ================================
# Carpeta MADRE que contiene las subcarpetas de cada rueda
CARPETA_MADRE = r"C:\Users\pedro\Desktop\Lhicarsa\MASCARAS_A_CLASIFICAR"

# Parámetros Geométricos
MAX_SIDE = 640
TOP_RATIO = 0.85 
GROUND_Y_PERCENTILE = 99.8

print("="*60)
print("🚀 INICIANDO PIPELINE END-TO-END: CLASIFICADOR DE EJES")
print("="*60)

# ================================
# 2. FUNCIONES BASE
# ================================
def extract_pressure(filename):
    patron = r'_((?:8\'5)|(?:7\'75)|7|(?:6\'25)|(?:5\'5)|(?:4\'75))_'
    match = re.search(patron, filename)
    if match:
        return float(match.group(1).replace("'", "."))
    return None

def resize_maintaining_aspect(image, max_side=640):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None

def fit_circle(points):
    x = points[:, 0]; y = points[:, 1]
    A = np.c_[2*x, 2*y, np.ones(len(points))]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(c0 + cx**2 + cy**2)
    return int(cx), int(cy), int(r)

def measure_flattening(mask, cx, cy, r):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0: return 0
    y_ground = int(np.percentile(ys, GROUND_Y_PERCENTILE))
    flattening_px = (cy + r) - y_ground
    return (flattening_px / r) * 100

# ================================
# 3. EXTRACCIÓN DE DATOS (VISIÓN ARTIFICIAL)
# ================================
datos_series = {}

print(f"\n🔍 Analizando carpetas en: {CARPETA_MADRE}...")

for root, dirs, files in os.walk(CARPETA_MADRE):
    # Saltamos la carpeta madre en sí, solo queremos las subcarpetas
    if root == CARPETA_MADRE: continue 
    
    nombre_subcarpeta = os.path.basename(root)
    presiones_serie = []
    achatamientos_serie = []

    for mask_name in files:
        if mask_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            
            pressure = extract_pressure(mask_name)
            if pressure is None: continue

            path_completo = os.path.join(root, mask_name)
            raw_mask = cv2.imread(path_completo, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None: continue

            mask_small = resize_maintaining_aspect(raw_mask, MAX_SIDE)
            _, mask_bin = cv2.threshold(mask_small, 127, 255, cv2.THRESH_BINARY)

            contour = largest_contour(mask_bin)
            if contour is None: continue

            # Puntos superiores para el círculo
            pts = contour.reshape(-1, 2)
            ys = pts[:, 1]
            limit = ys.min() + TOP_RATIO * (ys.max() - ys.min())
            pts_top = pts[ys <= limit]
            
            if len(pts_top) < 10: continue

            try:
                cx, cy, r = fit_circle(pts_top)
                flat_pct = measure_flattening(mask_bin, cx, cy, r)
                
                presiones_serie.append(pressure)
                achatamientos_serie.append(flat_pct)
            except:
                pass

    # Solo guardamos la serie si logró leer varias presiones para poder hacer la curva
    if len(presiones_serie) >= 3:
        datos_series[nombre_subcarpeta] = {
            "presiones": presiones_serie,
            "achatamientos": achatamientos_serie
        }
        print(f"✅ {nombre_subcarpeta}: Leídas {len(presiones_serie)} fotos válidas.")
    else:
        print(f"⚠️ {nombre_subcarpeta}: Ignorada (Insuficientes datos válidos).")

# ================================
# 4. CLASIFICACIÓN (MACHINE LEARNING)
# ================================
if len(datos_series) < 2:
    print("\n❌ Error: Necesitas al menos 2 subcarpetas válidas para hacer grupos.")
    exit()

nombres_series = []
matriz_features = []

for serie, datos in datos_series.items():
    x = np.array(datos["presiones"])
    y = np.array(datos["achatamientos"])
    
    # Huella Genética 1: Cuánto se aplasta en promedio
    achatamiento_medio = np.mean(y)
    
    # Huella Genética 2: Tasa de caída (Pendiente)
    pendiente, _ = np.polyfit(x, y, 1) 
    
    nombres_series.append(serie)
    matriz_features.append([achatamiento_medio, abs(pendiente)])

matriz_features = np.array(matriz_features)

# El cerebro de la IA que separa en 2 ejes
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
etiquetas = kmeans.fit_predict(matriz_features)

# ================================
# 5. INFORME FINAL Y VISUALIZACIÓN
# ================================
print("\n" + "="*60)
print("🎯 RESULTADO DE LA CLASIFICACIÓN DE EJES")
print("="*60)

grupo_0 = []
grupo_1 = []

for i, serie in enumerate(nombres_series):
    if etiquetas[i] == 0:
        grupo_0.append(serie)
    else:
        grupo_1.append(serie)

print("\n🔵 GRUPO 0 (Eje Tipo A):")
for s in grupo_0: print(f"   - {s}")

print("\n🔴 GRUPO 1 (Eje Tipo B):")
for s in grupo_1: print(f"   - {s}")

# --- GRÁFICAS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Análisis Inteligente de Deformación por Ejes", fontsize=16, fontweight='bold')

colores_grupos = {0: 'blue', 1: 'red'}

# Gráfica 1: Las curvas de deformación
for serie, datos in datos_series.items():
    # Encontrar a qué grupo pertenece esta serie
    idx = nombres_series.index(serie)
    grupo = etiquetas[idx]
    color = colores_grupos[grupo]
    
    # Ordenar puntos para que la línea se dibuje bien
    pts = sorted(zip(datos["presiones"], datos["achatamientos"]))
    x_val = [p[0] for p in pts]
    y_val = [p[1] for p in pts]
    
    ax1.plot(x_val, y_val, marker='o', linestyle='-', color=color, alpha=0.7, label=f"Grupo {grupo}" if f"Grupo {grupo}" not in ax1.get_legend_handles_labels()[1] else "")

ax1.set_title("Curvas de Achatamiento", fontsize=14)
ax1.set_xlabel("Presión (bar)", fontsize=12)
ax1.set_ylabel("Achatamiento (%)", fontsize=12)
ax1.invert_xaxis() # Invertimos para que se lea como "Pérdida de presión" de derecha a izquierda
ax1.grid(True, linestyle='--')
ax1.legend()

# Gráfica 2: El mapa mental del algoritmo K-Means
for i in range(len(nombres_series)):
    ax2.scatter(matriz_features[i, 0], matriz_features[i, 1], 
                color=colores_grupos[etiquetas[i]], s=150, edgecolors='black')
    ax2.text(matriz_features[i, 0], matriz_features[i, 1] + 0.1, nombres_series[i], fontsize=9)

ax2.set_title("Separación Matemática (K-Means)", fontsize=14)
ax2.set_xlabel("Achatamiento Medio", fontsize=12)
ax2.set_ylabel("Tasa de Caída (Pendiente)", fontsize=12)
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()