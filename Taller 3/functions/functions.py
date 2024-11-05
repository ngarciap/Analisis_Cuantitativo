import pandas as pd
import requests

def get_dataframe(url):
    # Descargar el contenido del archivo
    response = requests.get(url)
    data_text = response.text

    # Dividir el contenido en líneas
    lines = data_text.splitlines()

    # Saltar las primeras líneas que contienen la descripción y capturar solo los datos
    # En este caso, los datos empiezan en la línea 23 aproximadamente (ajustar si es necesario)
    data_lines = lines[22:]

    # Los datos están organizados en dos filas por cada entrada
    # Procesamos dos líneas a la vez para formar una fila de datos completa
    data = []
    for i in range(0, len(data_lines), 2):
        line1 = data_lines[i].strip()
        line2 = data_lines[i + 1].strip() if i + 1 < len(data_lines) else ''
        
        # Dividir ambas líneas en valores y combinarlos
        values = line1.split() + line2.split()
        data.append(values)

    # Crear un DataFrame y asignar nombres de columnas
    columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
            "PTRATIO", "B", "LSTAT", "MEDV"]
    df = pd.DataFrame(data, columns=columns)

    # Convertir las columnas a tipo numérico
    df = df.apply(pd.to_numeric)

    # Mostrar las primeras filas del DataFrame
    return df