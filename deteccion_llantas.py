import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np


def detectar_llanta(img):
    """
    Detección mejorada de la llanta metálica.

    Estrategia:
    1. Usar HoughCircles con parámetros ajustados para detectar el BORDE del aro metálico
    2. Validar con segmentación de color: la llanta es gris brillante (alto V, bajo S)
    3. Los agujeros de la llanta son oscuros dentro del aro -> validación adicional

    La clave es que el aro tiene un radio menor que el neumático completo.
    Para estas llantas de camión, el aro suele ser ~50-60% del radio total de la rueda.
    """
    alto, ancho = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Pre-procesamiento
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(gray_eq, (9, 9), 0)

    # --- Paso 1: Segmentar zona metálica de la llanta ---
    # La llanta cromada/metálica: saturación baja, valor alto
    h, s, v = cv2.split(img_hsv)

    # Máscara de metal brillante: S < 60, V > 130
    mask_metal = cv2.inRange(img_hsv, (0, 0, 130), (180, 60, 255))

    # Limpiar
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_metal = cv2.morphologyEx(
        mask_metal, cv2.MORPH_OPEN, kernel_small, iterations=1
    )
    mask_metal = cv2.morphologyEx(mask_metal, cv2.MORPH_CLOSE, kernel_med, iterations=2)

    # --- Paso 2: Encontrar el blob metálico más grande y circular ---
    contornos, _ = cv2.findContours(
        mask_metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mejor_llanta = None
    mejor_score = 0

    for c in contornos:
        area = cv2.contourArea(c)
        # Filtrar por tamaño: la llanta ocupa entre 1% y 15% de la imagen
        area_ratio = area / (alto * ancho)
        if area_ratio < 0.01 or area_ratio > 0.15:
            continue

        # Verificar circularidad
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidad = 4 * np.pi * area / (perimetro**2)

        if circularidad < 0.2:
            continue

        # Ajustar círculo mínimo circunscrito
        (mc_x, mc_y), mc_r = cv2.minEnclosingCircle(c)

        # Verificar que el área del contorno es similar al área del círculo
        area_circulo = np.pi * mc_r * mc_r
        llenado = area / area_circulo if area_circulo > 0 else 0

        # La llanta con agujeros tendrá llenado ~0.5-0.8 (no es sólida)
        if llenado < 0.25 or llenado > 0.95:
            continue

        # Score combinado
        score = circularidad * 0.4 + llenado * 0.3 + min(area_ratio * 20, 1) * 0.3

        if score > mejor_score:
            mejor_score = score
            mejor_llanta = (int(mc_x), int(mc_y), int(mc_r), score, "contorno")

    # --- Paso 3: HoughCircles como complemento/alternativa ---
    # Rango de radios: la llanta de camión suele ser entre 6% y 15% del ancho
    min_r = int(ancho * 0.06)
    max_r = int(ancho * 0.18)

    # Intentar con diferentes sensibilidades
    for param2 in [90, 70, 50]:
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(ancho * 0.2),
            param1=120,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is not None:
            for hx, hy, hr in np.round(circles[0]).astype(int):
                # Validar: dentro de este círculo, ¿hay metal brillante?
                mask_circulo = np.zeros_like(gray)
                cv2.circle(mask_circulo, (hx, hy), hr, 255, -1)
                metal_en_circulo = cv2.bitwise_and(mask_metal, mask_circulo)
                ratio_metal = (
                    np.sum(metal_en_circulo > 0) / (np.pi * hr * hr) if hr > 0 else 0
                )

                # La llanta con agujeros debe tener entre 30% y 80% de metal dentro del círculo
                if 0.20 < ratio_metal < 0.85:
                    # Verificar que hay zonas oscuras dentro (agujeros)
                    zona = gray_blur[
                        max(0, hy - hr) : min(alto, hy + hr),
                        max(0, hx - hr) : min(ancho, hx + hr),
                    ]
                    if zona.size > 0:
                        # Debe haber variación (metal + agujeros)
                        std_zona = np.std(zona)
                        if std_zona > 20:
                            score_h = (
                                ratio_metal * 0.4 + min(std_zona / 80, 1) * 0.3 + 0.3
                            )

                            # Si coincide con el contorno, bonus
                            if mejor_llanta is not None:
                                dist = np.sqrt(
                                    (hx - mejor_llanta[0]) ** 2
                                    + (hy - mejor_llanta[1]) ** 2
                                )
                                if dist < max_r * 0.5:
                                    score_h += 0.5

                            if mejor_llanta is None or score_h > mejor_llanta[3]:
                                mejor_llanta = (hx, hy, hr, score_h, "hough")
            break  # Si encontramos círculos, no necesitamos probar con menos sensibilidad

    # --- Paso 4: Si tenemos detección por contorno, refinar con Hough ---
    if mejor_llanta is not None and mejor_llanta[4] == "contorno":
        # Intentar refinar con HoughCircles en la zona del contorno
        cx_c, cy_c, r_c = mejor_llanta[0], mejor_llanta[1], mejor_llanta[2]
        margen = int(r_c * 0.5)
        x1 = max(0, cx_c - r_c - margen)
        y1 = max(0, cy_c - r_c - margen)
        x2 = min(ancho, cx_c + r_c + margen)
        y2 = min(alto, cy_c + r_c + margen)

        roi = gray_blur[y1:y2, x1:x2]
        if roi.shape[0] > 50 and roi.shape[1] > 50:
            circles_roi = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=50,
                param1=100,
                param2=60,
                minRadius=int(r_c * 0.7),
                maxRadius=int(r_c * 1.3),
            )
            if circles_roi is not None:
                hx, hy, hr = np.round(circles_roi[0, 0]).astype(int)
                mejor_llanta = (hx + x1, hy + y1, hr, mejor_llanta[3] + 0.2, "refinado")

    if mejor_llanta is None:
        return None

    return mejor_llanta[:4]  # (cx, cy, radio, confianza)


def detectar_contorno_neumatico(img, cx, cy, r_llanta, debug_img=None):
    """
    Detección mejorada del contorno del neumático.

    Mejoras:
    - Combina gradiente + segmentación + Canny adaptativo
    - Filtra por continuidad (elimina detecciones espurias del fondo)
    - Usa la geometría esperada del neumático como prior
    """
    alto, ancho = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # El neumático es negro/gris oscuro: V bajo a medio, bajo en S
    # El fondo (suelo de hormigón, carrocería, estanterías) es más claro o tiene otro color
    h, s, v = cv2.split(img_hsv)

    # Máscara de neumático: objetos oscuros
    mask_oscuro = (v < 90).astype(np.uint8) * 255

    # Limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_neumatico = cv2.morphologyEx(
        mask_oscuro, cv2.MORPH_CLOSE, kernel, iterations=3
    )
    mask_neumatico = cv2.morphologyEx(
        mask_neumatico, cv2.MORPH_OPEN, kernel, iterations=1
    )

    # Canny adaptativo
    mediana = np.median(gray)
    canny_low = int(max(0, 0.5 * mediana))
    canny_high = int(min(255, 1.3 * mediana))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), canny_low, canny_high)

    # --- Perfil radial ---
    num_angulos = 72  # Cada 5 grados
    perfil_radial = {}

    # Radio esperado del neumático: entre 1.4x y 2.5x el radio de la llanta
    r_min_neumatico = int(r_llanta * 1.3)
    r_max_neumatico = int(r_llanta * 2.8)

    for i in range(num_angulos):
        angulo = (2 * np.pi * i) / num_angulos
        angulo_grados = int(np.degrees(angulo))

        # Muestrear a lo largo del rayo
        intensidades = []
        es_neumatico = []
        es_borde = []

        for r in range(r_min_neumatico, r_max_neumatico):
            px = int(cx + r * np.cos(angulo))
            py = int(cy + r * np.sin(angulo))

            if 0 <= px < ancho and 0 <= py < alto:
                intensidades.append((r, gray[py, px]))
                es_neumatico.append(mask_neumatico[py, px] > 0)
                es_borde.append(edges[py, px] > 0)
            else:
                break

        if len(intensidades) < 10:
            continue

        # Estrategia: buscar donde termina el neumático
        # 1. Buscar último pixel consecutivo que es "neumático" (oscuro)
        # 2. Validar con bordes Canny

        # Método combinado: ventana deslizante
        ventana = 7
        mejor_r = r_max_neumatico
        encontrado = False

        for j in range(ventana, len(intensidades) - ventana):
            r_actual = intensidades[j][0]

            # Contar píxeles de neumático antes y después
            neumatico_antes = sum(es_neumatico[max(0, j - ventana) : j])
            neumatico_despues = sum(
                es_neumatico[j : min(len(es_neumatico), j + ventana)]
            )

            # Gradiente de brillo
            brillo_antes = np.mean(
                [intensidades[k][1] for k in range(max(0, j - ventana), j)]
            )
            brillo_despues = np.mean(
                [
                    intensidades[k][1]
                    for k in range(j, min(len(intensidades), j + ventana))
                ]
            )

            # Transición: de mayormente neumático a mayormente no-neumático
            # Y/o salto significativo de brillo
            if neumatico_antes >= ventana * 0.5 and neumatico_despues <= ventana * 0.3:
                # Hay un borde Canny cerca?
                tiene_borde = any(es_borde[max(0, j - 3) : min(len(es_borde), j + 3)])
                if tiene_borde or (brillo_despues - brillo_antes > 20):
                    mejor_r = r_actual
                    encontrado = True
                    break

            # Alternativa: salto brusco de brillo sin máscara
            if not encontrado and (brillo_despues - brillo_antes > 35):
                if any(es_borde[max(0, j - 2) : min(len(es_borde), j + 2)]):
                    mejor_r = r_actual
                    encontrado = True
                    break

        # Si no encontramos transición clara, buscar el último borde Canny
        if not encontrado:
            for j in range(len(es_borde) - 1, ventana, -1):
                if es_borde[j] and es_neumatico[max(0, j - 3)]:
                    mejor_r = intensidades[j][0]
                    break

        perfil_radial[angulo_grados] = mejor_r

        # Debug: dibujar puntos del contorno
        if debug_img is not None:
            px = int(cx + mejor_r * np.cos(angulo))
            py = int(cy + mejor_r * np.sin(angulo))
            if 0 <= px < ancho and 0 <= py < alto:
                color = (0, 255, 255) if encontrado else (0, 100, 200)
                cv2.circle(debug_img, (px, py), 3, color, -1)

    # --- Post-procesamiento: filtrar outliers del perfil ---
    if len(perfil_radial) > 10:
        radios = list(perfil_radial.values())
        mediana_r = np.median(radios)
        iqr = np.percentile(radios, 75) - np.percentile(radios, 25)

        # Eliminar puntos que están muy lejos de la mediana (outliers del fondo)
        perfil_filtrado = {}
        for ang, r in perfil_radial.items():
            if abs(r - mediana_r) < iqr * 2.5:
                perfil_filtrado[ang] = r
            else:
                # Reemplazar outlier con la mediana
                perfil_filtrado[ang] = int(mediana_r)

        perfil_radial = perfil_filtrado

    return perfil_radial


def detectar_suelo(img, cx, cy, r_llanta):
    """
    Detecta el suelo buscando la transición neumático/hormigón debajo de la rueda.
    """
    alto, ancho = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buscar en una franja vertical centrada en la rueda
    franja_ancho = int(r_llanta * 0.5)
    x1 = max(0, cx - franja_ancho)
    x2 = min(ancho, cx + franja_ancho)

    # Desde debajo del centro de la llanta
    y_inicio = cy + int(r_llanta * 0.8)

    if y_inicio >= alto:
        return alto - 1

    # Perfil vertical promedio de brillo
    franja = gray[y_inicio:, x1:x2]
    if franja.size == 0:
        return alto - 1

    perfil_v = np.mean(franja, axis=1)

    # Suavizar
    kernel = np.ones(5) / 5
    if len(perfil_v) > 5:
        perfil_suav = np.convolve(perfil_v, kernel, mode="valid")

        # Buscar el primer salto significativo (oscuro->claro = neumático->suelo)
        gradiente = np.diff(perfil_suav)

        for j in range(len(gradiente)):
            if gradiente[j] > 15:  # Salto positivo de brillo
                return y_inicio + j + 2  # +2 por offset del suavizado

    return alto - 1


def calcular_metricas(cx, cy, r_llanta, perfil_radial, y_suelo):
    """
    Métricas refinadas normalizadas por radio de llanta.
    """
    metricas = {}

    # Distancia centro->suelo normalizada
    dist_suelo = abs(y_suelo - cy)
    metricas["dist_suelo_px"] = dist_suelo
    metricas["ratio_suelo"] = dist_suelo / r_llanta if r_llanta > 0 else 0

    if not perfil_radial:
        return metricas

    angulos = sorted(perfil_radial.keys())
    radios = np.array([perfil_radial[a] for a in angulos])

    # Radio promedio y normalizado
    radio_medio = np.mean(radios)
    metricas["radio_medio_px"] = float(radio_medio)
    metricas["radio_medio_norm"] = float(radio_medio / r_llanta) if r_llanta > 0 else 0

    # Desviación estándar (indica deformación general)
    metricas["std_radio"] = float(np.std(radios))
    metricas["std_norm"] = float(np.std(radios) / r_llanta) if r_llanta > 0 else 0

    # Análisis por cuadrantes (grados: 0=der, 90=abajo, 180=izq, 270=arriba)
    def radios_rango(a1, a2):
        """Obtiene radios en un rango angular"""
        if a1 < a2:
            vals = [perfil_radial[a] for a in angulos if a1 <= a <= a2]
        else:  # Cruce por 360
            vals = [perfil_radial[a] for a in angulos if a >= a1 or a <= a2]
        return vals if vals else [radio_medio]

    r_arriba = np.mean(radios_rango(250, 290))
    r_abajo = np.mean(radios_rango(70, 110))
    r_derecha = np.mean(radios_rango(340, 20))
    r_izquierda = np.mean(radios_rango(160, 200))

    metricas["r_arriba"] = float(r_arriba)
    metricas["r_abajo"] = float(r_abajo)
    metricas["r_derecha"] = float(r_derecha)
    metricas["r_izquierda"] = float(r_izquierda)

    # MÉTRICAS CLAVE para inferencia de presión:

    # 1. Ratio abajo/arriba - indica aplastamiento (< 1 = aplastado abajo)
    metricas["ratio_bottom_top"] = float(r_abajo / r_arriba) if r_arriba > 0 else 0

    # 2. Radio inferior normalizado por llanta - más directo
    metricas["r_abajo_norm"] = float(r_abajo / r_llanta) if r_llanta > 0 else 0

    # 3. Sidewall ratio - lateral vs inferior
    r_lateral = (r_derecha + r_izquierda) / 2
    metricas["sidewall_ratio"] = float(r_lateral / r_abajo) if r_abajo > 0 else 0

    # 4. Flat spot: ángulo de zona "aplastada"
    # La zona de contacto tiene radios significativamente menores
    umbral = radio_medio - np.std(radios) * 0.5
    n_aplastados = sum(1 for r in radios if r < umbral)
    metricas["flat_spot_deg"] = float(n_aplastados * (360 / len(angulos)))

    # 5. Deflexión: diferencia entre radio máximo y mínimo normalizada
    metricas["deflexion"] = (
        float((np.max(radios) - np.min(radios)) / r_llanta) if r_llanta > 0 else 0
    )

    # 6. Excentricidad del perfil (cuánto se desvía de un círculo perfecto)
    if len(radios) > 5:
        # Ajustar círculo ideal y medir desviación
        radio_ideal = np.mean(radios)
        residuos = radios - radio_ideal
        metricas["excentricidad"] = (
            float(np.std(residuos) / radio_ideal) if radio_ideal > 0 else 0
        )

    return metricas


def analizar_imagen(
    ruta_imagen,
    presion_real=None,
    guardar_debug=True,
    output_dir="/home/claude/resultados",
):
    """Pipeline de análisis v2."""
    img_original = cv2.imread(ruta_imagen)
    if img_original is None:
        return None, None

    # Redimensionar
    ancho_objetivo = 800
    alto_o, ancho_o = img_original.shape[:2]
    factor = ancho_objetivo / ancho_o
    img = cv2.resize(img_original, (ancho_objetivo, int(alto_o * factor)))
    alto_s, ancho_s = img.shape[:2]

    debug_img = img.copy()

    # 1. Detectar llanta
    deteccion = detectar_llanta(img)

    if deteccion is None:
        cv2.putText(
            debug_img,
            "LLANTA NO DETECTADA",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        if guardar_debug:
            nombre = Path(ruta_imagen).stem
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{nombre}_FALLO.jpg", debug_img)
        return debug_img, None

    cx, cy, r_llanta, confianza = deteccion

    # Dibujar llanta
    cv2.circle(debug_img, (cx, cy), r_llanta, (0, 255, 0), 2)
    cv2.circle(debug_img, (cx, cy), 4, (0, 255, 0), -1)

    # 2. Perfil del neumático
    perfil = detectar_contorno_neumatico(img, cx, cy, r_llanta, debug_img)

    # 3. Suelo
    y_suelo = detectar_suelo(img, cx, cy, r_llanta)
    cv2.line(debug_img, (0, y_suelo), (ancho_s, y_suelo), (0, 165, 255), 1)
    cv2.line(debug_img, (cx, cy), (cx, y_suelo), (255, 255, 0), 2)

    # 4. Métricas
    metricas = calcular_metricas(cx, cy, r_llanta, perfil, y_suelo)
    metricas["confianza"] = float(confianza)
    metricas["r_llanta_px"] = int(r_llanta)
    metricas["cx"] = int(cx)
    metricas["cy"] = int(cy)
    if presion_real is not None:
        metricas["presion_real"] = presion_real
    metricas["archivo"] = os.path.basename(ruta_imagen)

    # 5. Dibujar info
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_t = 22

    if presion_real is not None:
        cv2.putText(
            debug_img,
            f"Presion: {presion_real} bar",
            (10, y_t),
            font,
            0.65,
            (0, 255, 255),
            2,
        )
        y_t += 22

    cv2.putText(
        debug_img,
        f"Llanta r={r_llanta}px conf={confianza:.2f}",
        (10, y_t),
        font,
        0.45,
        (200, 200, 200),
        1,
    )
    y_t += 18

    for key in [
        "ratio_suelo",
        "ratio_bottom_top",
        "sidewall_ratio",
        "r_abajo_norm",
        "flat_spot_deg",
        "deflexion",
    ]:
        if key in metricas:
            cv2.putText(
                debug_img,
                f"{key}: {metricas[key]:.3f}",
                (10, y_t),
                font,
                0.42,
                (200, 200, 200),
                1,
            )
            y_t += 16

    # Dibujar perfil como polígono
    if perfil:
        pts = []
        for ang_g in sorted(perfil.keys()):
            ang_r = np.radians(ang_g)
            r = perfil[ang_g]
            px = int(cx + r * np.cos(ang_r))
            py = int(cy + r * np.sin(ang_r))
            if 0 <= px < ancho_s and 0 <= py < alto_s:
                pts.append([px, py])
        if len(pts) > 3:
            cv2.polylines(
                debug_img, [np.array(pts, dtype=np.int32)], True, (0, 150, 255), 1
            )

    # Guardar
    if guardar_debug:
        nombre = Path(ruta_imagen).stem
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/{nombre}.jpg", debug_img)

    return debug_img, metricas


def procesar_todo(carpeta, output_dir="/home/claude/resultados"):
    """Procesa todas las imágenes y genera análisis."""
    archivos = sorted(glob.glob(f"{carpeta}/*.JPG") + glob.glob(f"{carpeta}/*.jpg"))
    print(f"Encontradas {len(archivos)} imágenes\n")

    resultados = []

    for ruta in archivos:
        nombre = Path(ruta).stem
        partes = nombre.split("_")

        if len(partes) >= 6:
            angulo = partes[3]
            presion_str = partes[4]
            camion = partes[2]

            try:
                if len(presion_str) == 1:
                    presion = float(presion_str)
                elif len(presion_str) == 2:
                    presion = float(presion_str) / 10
                elif len(presion_str) == 3:
                    presion = float(presion_str) / 100
                else:
                    presion = float(presion_str)
            except ValueError:
                presion = None

            _, metricas = analizar_imagen(ruta, presion, True, output_dir)

            if metricas:
                metricas["angulo"] = angulo
                metricas["camion"] = camion
                resultados.append(metricas)
                print(
                    f"  ✓ {nombre} | P={presion} | r_llanta={metricas['r_llanta_px']} | ratio_suelo={metricas.get('ratio_suelo', 0):.3f}"
                )
            else:
                print(f"  ✗ {nombre} | FALLO detección")

    return resultados


def generar_graficas(resultados, output_dir="/home/claude/resultados"):
    """Genera gráficas comparativas presión vs métricas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = [
        r for r in resultados if "presion_real" in r and r["presion_real"] is not None
    ]
    if not datos:
        print("Sin datos")
        return

    angulos = sorted(set(d.get("angulo", "?") for d in datos))
    presiones = sorted(set(d["presion_real"] for d in datos))

    metricas_plot = [
        ("ratio_suelo", "Dist Centro→Suelo / R.Llanta", "Debe ↓ con menos presión"),
        ("ratio_bottom_top", "Radio Bottom / Radio Top", "Debe ↓ con menos presión"),
        ("r_abajo_norm", "Radio Inferior / R.Llanta", "Debe ↓ con menos presión"),
        ("sidewall_ratio", "Sidewall Ratio (Lat/Bottom)", "Debe ↑ con menos presión"),
        ("flat_spot_deg", "Flat Spot (grados)", "Debe ↑ con menos presión"),
        ("deflexion", "Deflexión (Max-Min)/R.Llanta", "Debe ↑ con menos presión"),
    ]

    for angulo_cam in angulos:
        datos_ang = [d for d in datos if d.get("angulo") == angulo_cam]
        if len(datos_ang) < 3:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Métricas de Deformación vs Presión — Vista: {angulo_cam}\n"
            f"(Llanta como referencia, n={len(datos_ang)} imágenes)",
            fontsize=13,
            fontweight="bold",
        )

        for idx, (metrica, titulo, nota) in enumerate(metricas_plot):
            ax = axes[idx // 3][idx % 3]

            vpor_presion = {}
            for d in datos_ang:
                p = d["presion_real"]
                if metrica in d:
                    vpor_presion.setdefault(p, []).append(d[metrica])

            if vpor_presion:
                ps = sorted(vpor_presion.keys())
                medias = [np.mean(vpor_presion[p]) for p in ps]
                stds = [np.std(vpor_presion[p]) for p in ps]

                # Puntos individuales
                for p in ps:
                    ax.scatter(
                        [p] * len(vpor_presion[p]),
                        vpor_presion[p],
                        alpha=0.3,
                        s=30,
                        color="#90CAF9",
                    )

                # Media con error bars
                ax.errorbar(
                    ps,
                    medias,
                    yerr=stds,
                    fmt="o-",
                    capsize=5,
                    linewidth=2,
                    markersize=8,
                    color="#1565C0",
                    zorder=5,
                )

                # Tendencia lineal
                if len(ps) > 2:
                    z = np.polyfit(ps, medias, 1)
                    x_fit = np.linspace(min(ps), max(ps), 100)
                    ax.plot(
                        x_fit,
                        np.polyval(z, x_fit),
                        "--",
                        color="red",
                        alpha=0.5,
                        label=f"pendiente={z[0]:.4f}",
                    )
                    ax.legend(fontsize=8)

                ax.set_xlabel("Presión (bar)")
                ax.set_ylabel(metrica)
                ax.set_title(f"{titulo}\n({nota})", fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            f"{output_dir}/grafica_{angulo_cam}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"  Guardada: grafica_{angulo_cam}.png")

    # Correlaciones
    print("\n" + "=" * 70)
    print("CORRELACIONES PRESIÓN vs MÉTRICAS")
    print("=" * 70)

    for angulo_cam in angulos:
        datos_ang = [d for d in datos if d.get("angulo") == angulo_cam]
        if len(datos_ang) < 4:
            continue

        print(f"\n--- Vista: {angulo_cam} ({len(datos_ang)} muestras) ---")

        presiones_arr = np.array([d["presion_real"] for d in datos_ang])

        for metrica, titulo, _ in metricas_plot:
            valores = np.array([d.get(metrica, np.nan) for d in datos_ang])
            mask_valid = ~np.isnan(valores)
            if np.sum(mask_valid) > 3:
                corr = np.corrcoef(presiones_arr[mask_valid], valores[mask_valid])[0, 1]
                print(
                    f"  {titulo:45s} r = {corr:+.4f}  {'⭐' if abs(corr) > 0.3 else '  '}"
                )

    # Guardar JSON
    datos_json = []
    for d in datos:
        d_clean = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in d.items()
        }
        datos_json.append(d_clean)

    with open(f"{output_dir}/metricas.json", "w") as f:
        json.dump(datos_json, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas guardadas en {output_dir}/metricas.json")


# ============================================================
if __name__ == "__main__":
    OUTPUT = r"./resultados"

    print("=" * 60)
    print("DETECCIÓN MEJORADA")
    print("Llanta como referencia + perfil radial robusto")
    print("=" * 60 + "\n")

    resultados = procesar_todo(r"./Lhicarsa_imagenes", OUTPUT)
    print(f"\n✅ {len(resultados)} imágenes procesadas")

    generar_graficas(resultados, OUTPUT)
