import cv2
import numpy as np
import pandas as pd
import os

def detectar_llanta_y_bordes(ruta_imagen, presion_real):
    # Cargar imagen
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        return None, False

    # Redimensionar (se supone que mejora la velocidad porque antes de añadir esto tardaba la vida)
    ancho_objetivo = 800
    alto, ancho = img_original.shape[:2]
    factor_escala = ancho_objetivo / ancho
    nuevas_dims = (ancho_objetivo, int(alto * factor_escala))
    img_small = cv2.resize(img_original, nuevas_dims)
    
    # Copia para visualizar
    output = img_small.copy()
    alto_s, ancho_s = output.shape[:2]

    # Pre-procesamiento para Hough
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (9, 9), 0)

    # Detectar Llanta con Hough 
    #! Funciona bien exceptuando imagenes con doble rueda
    circles = cv2.HoughCircles(
        gray_blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=120,
        param2=100, 
        minRadius=50,
        maxRadius=150 
    )

    if circles is None:
        return output, False

    # Procesar la mejor detección
    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    cx, cy, r_llanta = circles[0]

    # Dibujar la llanta detectada 
    cv2.circle(output, (cx, cy), r_llanta, (0, 255, 0), 2)
    cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)

    # Intento de detección de bordes del neumático
    
    # Usamos Canny para encontrar todos los bordes de la imagen
    edges = cv2.Canny(gray_blur, 50, 150)
    
    # Margen de seguridad: Empezamos a buscar bordes un poco más allá 
    # del radio de la llanta detectada (ej. radio * 1.2) para no confundirnos con el borde del metal.
    inicio_busqueda = int(r_llanta * 1.3)
    
    y_top = 0
    y_bottom = alto_s - 1

    # --- Búsqueda Hacia ARRIBA (Top) ---
    # Recorremos desde el borde de la llanta hacia arriba
    # Buscamos en una franja vertical pequeña (cx-2 a cx+2) para reducir ruido
    for y in range(cy - inicio_busqueda, 0, -1):
        # Si encontramos un píxel de borde en la columna central
        if edges[y, cx] > 0 or edges[y, cx-2] > 0 or edges[y, cx+2] > 0:
            y_top = y
            break # Encontramos el primer borde externo
            
    # --- Búsqueda Hacia ABAJO (Bottom) ---
    for y in range(cy + inicio_busqueda, alto_s):
        if edges[y, cx] > 0 or edges[y, cx-2] > 0 or edges[y, cx+2] > 0:
            y_bottom = y
            break
    
    # Aun con la reducción de ruido sigue habiendo errores de detección tanto en top como en bottom

    # Cálculos y Visualización
    dist_top = abs(cy - y_top)
    dist_bot = abs(cy - y_bottom)
    
    # Dibujar líneas de medición
    cv2.line(output, (cx, cy), (cx, y_top), (255, 0, 0), 2)   # Línea Azul (Arriba)
    cv2.line(output, (cx - 20, y_top), (cx + 20, y_top), (255, 0, 0), 2) # Tope
    
    cv2.line(output, (cx, cy), (cx, y_bottom), (0, 255, 255), 2) # Línea Amarilla (Abajo)
    cv2.line(output, (cx - 20, y_bottom), (cx + 20, y_bottom), (0, 255, 255), 2) # Tope

    # Calcular relación (Ratio)
    # Si ratio < 1.0, la parte de abajo es más corta (rueda aplastada)
    #* Nota: A veces la cámara está más alta y la perspectiva engaña, 
    #* con lo que he pensado que un posible paso es hacer algún tipo de normalización
    ratio = dist_bot / dist_top if dist_top > 0 else 0

    # Mostrar Textos en Pantalla
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Presión (Dato del CSV)
    cv2.putText(output, f"Presion: {presion_real}", (10, 30), font, 0.8, (0, 255, 255), 2)
    
    # Distancias en píxeles
    texto_top = f"D.Top: {dist_top} px"
    texto_bot = f"D.Bot: {dist_bot} px"
    cv2.putText(output, texto_top, (cx + 20, cy - 40), font, 0.6, (255, 100, 100), 2)
    cv2.putText(output, texto_bot, (cx + 20, cy + 60), font, 0.6, (100, 255, 255), 2)
    
    # Resultado del análisis
    cv2.putText(output, f"Ratio Bot/Top: {ratio:.3f}", (10, alto_s - 20), font, 0.7, (255, 255, 255), 2)

    return output, True

# --- Bloque de prueba del código ---
csv_path = 'dataset_ruedas.csv'
df = pd.read_csv(csv_path)
df_frontal = df[df['Angulo'].str.strip() == 'Frontal']

print(f"Procesando {len(df_frontal)} imágenes frontales...")


for index, row in df_frontal.head(36).iterrows(): # El head es simplemente para tomar mas o menos muestras
    ruta_original = row['Ruta_Imagen']
    presion = row['Presion']
    
    if os.path.exists(ruta_original):
        print(f"Procesando: Presión {presion} - {os.path.basename(ruta_original)}")
        
        imagen_res, exito = detectar_llanta_y_bordes(ruta_original, presion)
        
        if exito:
            cv2.imshow("Analisis de Deformacion", imagen_res)
            # Pausa hasta presionar tecla. ESC para salir.
            key = cv2.waitKey(0) 
            if key == 27: 
                break
    else:
        print(f"No existe: {ruta_original}")

cv2.destroyAllWindows()