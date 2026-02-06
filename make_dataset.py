import pandas as pd
from pathlib import Path


ruta_carpeta = Path("./Lhicarsa_imagenes") 

datos = []


extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for archivo in ruta_carpeta.iterdir():
    if archivo.suffix.lower() in extensiones_validas:
        
        nombre_sin_ext = archivo.stem
        
        
        partes = nombre_sin_ext.split('_')
        
        
        if len(partes) == 6:
            tipo, camion, rueda, angulo, presion, fecha_str = partes
            
            datos.append({
                "TipoCamion": tipo,
                "Camion": camion,
                "Rueda": rueda,
                "Angulo": angulo,
                "Presion": presion,
                "Fecha_Original": fecha_str,
                "Ruta_Imagen": str(archivo.absolute()) 
            })

df = pd.DataFrame(datos)

if not df.empty:
    df['Fecha'] = pd.to_datetime(df['Fecha_Original'], format='%d-%m-%Y', errors='coerce')
    
    columnas = ["TipoCamion", "Camion", "Rueda", "Angulo", "Presion", "Fecha", "Ruta_Imagen"]
    df = df[columnas]

print(f"Se han procesado {len(df)} imágenes.")
print(df.head())

df.to_csv("dataset_ruedas.csv", index=False)