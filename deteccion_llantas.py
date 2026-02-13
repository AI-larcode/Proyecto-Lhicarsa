import cv2
import numpy as np
import pandas as pd
import os

def detectar_llanta_y_bordes(ruta_imagen, presion_real):
    # 1. Cargar y Redimensionar
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        return None, False, None

    ancho_objetivo = 800
    alto, ancho = img_original.shape[:2]
    factor_escala = ancho_objetivo / ancho
    nuevas_dims = (ancho_objetivo, int(alto * factor_escala))
    img_small = cv2.resize(img_original, nuevas_dims)
    
    output = img_small.copy()
    alto_s, ancho_s = output.shape[:2]

    # 2. Pre-procesamiento mejorado
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (9, 9), 0)

    # 3. Detección de círculo (llanta)
    circles = cv2.HoughCircles(
        gray_blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=120,
        param2=100, 
        minRadius=50,
        maxRadius=100 
        # Esto funciona en la mayoria de imagenes pero en algunas con mucha definición/calidad no
        #! Buscar solución (posiblemente un redimensionado de las imagenes con mayor definición/calidad)
    )

    if circles is None:
        return output, False, None

    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    cx, cy, r_llanta = circles[0]

    cv2.circle(output, (cx, cy), r_llanta, (0, 255, 0), 2)
    cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
    
    # 4. Detección de bordes mejorada
    # Usar múltiples estrategias de edge detection
    edges_canny = cv2.Canny(gray_blur, 30, 100)
    
    # Sobel para detectar gradientes horizontales (bordes horizontales del neumático)
    sobelx = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    
    # Combinar detecciones
    edges_combined = cv2.bitwise_or(edges_canny, cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)[1])
    
    # 5. Parámetros de búsqueda dinámicos
    ancho_franja = int(r_llanta * 0.4)  # Franja más ancha
    
    # Rango de búsqueda más inteligente
    inicio_busqueda = int(r_llanta * 1.1)
    fin_busqueda = int(r_llanta * 2.5)
    
    # 6. Función de búsqueda mejorada con múltiples criterios
    def encontrar_borde_neumatico(img_edges, cx, cy_start, direccion, r_llanta, ancho_franja):
        """
        direccion: 1 para abajo, -1 para arriba
        """
        mejor_y = cy_start
        max_score = 0
        candidatos = []
        
        step = direccion
        y_inicio = cy_start + (inicio_busqueda * direccion)
        y_fin = cy_start + (fin_busqueda * direccion)
        
        y_range = range(y_inicio, y_fin, step) if direccion > 0 else range(y_inicio, y_fin, step)
        
        for y in y_range:
            if y < 5 or y >= img_edges.shape[0] - 5:
                continue
            
            # Región de análisis
            x1 = max(0, cx - ancho_franja)
            x2 = min(img_edges.shape[1], cx + ancho_franja)
            
            # Calcular múltiples métricas
            region = img_edges[y-2:y+3, x1:x2]  # Región de 5 píxeles de alto
            
            if region.size == 0:
                continue
            
            # Métrica 1: Energía de bordes
            energia = np.sum(region > 0)
            
            # Métrica 2: Continuidad horizontal (borde debe ser continuo)
            fila_central = region[2, :]
            continuidad = np.sum(fila_central > 0) / max(1, len(fila_central))
            
            # Métrica 3: Contraste (debe haber diferencia clara)
            if y + 10 < img_edges.shape[0] and y - 10 >= 0:
                region_arriba = img_edges[y-10:y-5, x1:x2]
                region_abajo = img_edges[y+5:y+10, x1:x2]
                contraste = abs(np.mean(region_arriba) - np.mean(region_abajo))
            else:
                contraste = 0
            
            # Score combinado (ponderado)
            score = (energia * 0.4) + (continuidad * 100 * 0.4) + (contraste * 0.2)
            
            candidatos.append({
                'y': y,
                'score': score,
                'energia': energia,
                'continuidad': continuidad
            })
            
            if score > max_score and continuidad > 0.3:  # Mínimo de continuidad
                max_score = score
                mejor_y = y
        
        # Validación: el borde debe estar fuera de la llanta
        distancia_desde_centro = abs(mejor_y - cy_start)
        if distancia_desde_centro < r_llanta * 0.8:
            # Buscar el siguiente mejor candidato que esté más lejos
            candidatos_validos = [c for c in candidatos 
                                 if abs(c['y'] - cy_start) >= r_llanta * 0.8]
            if candidatos_validos:
                mejor_candidato = max(candidatos_validos, key=lambda x: x['score'])
                mejor_y = mejor_candidato['y']
        
        return mejor_y
    
    # 7. Búsqueda de bordes superior e inferior
    y_top = encontrar_borde_neumatico(edges_combined, cx, cy, -1, r_llanta, ancho_franja)
    y_bottom = encontrar_borde_neumatico(edges_combined, cx, cy, 1, r_llanta, ancho_franja)
    
    # 8. Cálculos y visualización
    dist_top = abs(cy - y_top)
    dist_bot = abs(cy - y_bottom)
    
    # Dibujar mediciones
    cv2.line(output, (cx, cy), (cx, y_top), (255, 0, 0), 2) 
    cv2.line(output, (cx - 30, y_top), (cx + 30, y_top), (255, 0, 0), 3)
    
    cv2.line(output, (cx, cy), (cx, y_bottom), (0, 255, 255), 2) 
    cv2.line(output, (cx - 30, y_bottom), (cx + 30, y_bottom), (0, 255, 255), 3)
    
    # Normalización
    ratio_top = dist_top / r_llanta
    ratio_bot = dist_bot / r_llanta
    
    # Métricas múltiples
    ratio_deformacion = ratio_bot / ratio_top if ratio_top > 0 else 0
    suma_total = ratio_top + ratio_bot  # Diámetro total normalizado
    diferencia_abs = abs(ratio_bot - ratio_top)  # Asimetría absoluta
    
    # Textos
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, f"Presion: {presion_real} bar", (10, 30), font, 0.8, (0, 255, 255), 2)
    
    cv2.putText(output, f"D.Top: {ratio_top:.2f}R", (cx + 40, y_top), font, 0.6, (255, 100, 100), 2)
    cv2.putText(output, f"D.Bot: {ratio_bot:.2f}R", (cx + 40, y_bottom), font, 0.6, (100, 255, 255), 2)
    
    cv2.putText(output, f"Ratio: {ratio_deformacion:.3f}", (10, alto_s - 60), font, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Suma: {suma_total:.2f}R", (10, alto_s - 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Asim: {diferencia_abs:.3f}R", (10, alto_s - 0), font, 0.7, (255, 255, 255), 2)
    
    # Datos para análisis
    metricas = {
        'presion': presion_real,
        'ratio_top': ratio_top,
        'ratio_bot': ratio_bot,
        'ratio_deformacion': ratio_deformacion,
        'suma_total': suma_total,
        'asimetria': diferencia_abs,
        'r_llanta': r_llanta
    }
    
    return output, True, metricas

# --- Bloque de análisis ---
csv_path = 'dataset_ruedas.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df_frontal = df[df['Angulo'].str.strip() == 'Frontal']

    print(f"Procesando {len(df_frontal)} imágenes frontales...")
    
    resultados = []

    for index, row in df_frontal.head(36).iterrows():
        ruta_original = row['Ruta_Imagen']
        presion = row['Presion']
        
        if os.path.exists(ruta_original):
            print(f"Procesando: Presión {presion} - {os.path.basename(ruta_original)}")
            
            imagen_res, exito, metricas = detectar_llanta_y_bordes(ruta_original, presion)
            
            if exito and metricas:
                resultados.append(metricas)
                
                cv2.imshow("Analisis de Deformacion", imagen_res)
                key = cv2.waitKey(0)
                if key == 27: 
                    break
        else:
            print(f"No existe: {ruta_original}")

    cv2.destroyAllWindows()
    
    # Análisis de correlaciones
    #! Esta parte del código falla
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        print("\n=== ANÁLISIS DE CORRELACIONES ===")
        print(df_resultados[['presion', 'ratio_deformacion', 'suma_total', 'asimetria']].corr())
        print("\n=== ESTADÍSTICAS POR MÉTRICA ===")
        print(df_resultados.describe())
        
        # Guardar resultados
        df_resultados.to_csv('resultados_analisis.csv', index=False)
        print("\nResultados guardados en 'resultados_analisis.csv'")
        
else:
    print("CSV no encontrado.")