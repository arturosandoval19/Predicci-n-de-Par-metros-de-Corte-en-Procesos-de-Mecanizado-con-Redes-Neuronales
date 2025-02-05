import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

def cargar_datos(archivo):
    """
    Cargar datos desde un archivo CSV con múltiples intentos.
    """
    if not os.path.exists(archivo):
        print("Archivo no encontrado. Por favor, verifica el nombre y ubicación.")
        return None

    # Intentar diferentes delimitadores y configuraciones
    delimitadores = ['\t', ';', ',', ' ']
    
    for delimitador in delimitadores:
        try:
            # Leer con diferentes opciones
            datos = pd.read_csv(archivo, 
                                delimiter=delimitador, 
                                header=0,  # Asumir primera fila como encabezado
                                encoding='utf-8',
                                skipinitialspace=True)
            
            # Imprimir columnas detectadas para depuración
            print("Columnas detectadas:", list(datos.columns))
            
            # Verificar si tiene las columnas necesarias
            columnas_requeridas = ['Vc', 't', 'f', 'Ra']
            
            # Intentar hacer mapeo de columnas si no coinciden exactamente
            mapeo_columnas = {}
            for col_req in columnas_requeridas:
                # Buscar columna que contenga el nombre (insensible a mayúsculas)
                coincidencias = [col for col in datos.columns if col_req.lower() in col.lower()]
                if coincidencias:
                    mapeo_columnas[coincidencias[0]] = col_req
            
            # Renombrar columnas si se encontraron coincidencias
            if mapeo_columnas:
                datos = datos.rename(columns=mapeo_columnas)
            
            # Verificar que todas las columnas necesarias estén presentes
            if all(col in datos.columns for col in columnas_requeridas):
                print(f"Se cargaron {len(datos)} registros del archivo.")
                
                # Convertir columnas a numérico, ignorando errores
                for col in columnas_requeridas:
                    datos[col] = pd.to_numeric(datos[col], errors='coerce')
                
                # Eliminar filas con valores NaN
                datos = datos.dropna(subset=columnas_requeridas)
                
                return datos
            
        except Exception as e:
            print(f"Error al intentar cargar con delimitador '{delimitador}': {e}")
    
    print("No se pudo cargar el archivo con ningún método.")
    return None

def buscar_coincidencia_exacta(Ra, datos, tolerancia=1e-3):
    """
    Buscar coincidencia exacta para un valor de Ra con tolerancia.
    """
    # Encontrar la fila más cercana
    coincidencia = datos[np.abs(datos['Ra'] - Ra) < tolerancia]
    
    if not coincidencia.empty:
        # Tomar la primera coincidencia
        fila = coincidencia.iloc[0]
        return fila['Vc'], fila['t'], fila['f']
    
    return None, None, None

def preparar_datos(datos):
    """
    Preparar datos para entrenamiento de redes neuronales.
    """
    # Separar características y objetivos
    X = datos[['Ra']].values
    y_Vc = datos['Vc'].values
    y_t = datos['t'].values
    y_f = datos['f'].values
    
    # Dividir datos
    X_train, X_test, y_Vc_train, y_Vc_test, y_t_train, y_t_test, y_f_train, y_f_test = train_test_split(
        X, y_Vc, y_t, y_f, test_size=0.2, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, 
            y_Vc_train, y_Vc_test, 
            y_t_train, y_t_test, 
            y_f_train, y_f_test, 
            scaler)

def crear_modelo_keras():
    """
    Crear modelo de red neuronal con Keras.
    """
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    modelo.compile(optimizer='adam', loss='mse')
    return modelo

def entrenar_modelos_keras(datos):
    """
    Entrenar modelos de red neuronal para Vc, t, y f.
    """
    # Preparar datos
    (X_train, X_test, 
     y_Vc_train, y_Vc_test, 
     y_t_train, y_t_test, 
     y_f_train, y_f_test, 
     scaler) = preparar_datos(datos)
    
    # Crear y entrenar modelos
    modelo_Vc = crear_modelo_keras()
    modelo_t = crear_modelo_keras()
    modelo_f = crear_modelo_keras()
    
    # Early Stopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Entrenamiento con validación
    modelo_Vc.fit(X_train, y_Vc_train, validation_data=(X_test, y_Vc_test), epochs=500, batch_size=32, verbose=0, callbacks=[early_stopping])
    modelo_t.fit(X_train, y_t_train, validation_data=(X_test, y_t_test), epochs=500, batch_size=32, verbose=0, callbacks=[early_stopping])
    modelo_f.fit(X_train, y_f_train, validation_data=(X_test, y_f_test), epochs=500, batch_size=32, verbose=0, callbacks=[early_stopping])
    
    # Evaluar modelos
    print("Modelo Vc - Pérdida en test:", modelo_Vc.evaluate(X_test, y_Vc_test, verbose=0))
    print("Modelo t - Pérdida en test:", modelo_t.evaluate(X_test, y_t_test, verbose=0))
    print("Modelo f - Pérdida en test:", modelo_f.evaluate(X_test, y_f_test, verbose=0))
    
    return modelo_Vc, modelo_t, modelo_f, scaler

def predecir_valores(modelos, scaler, Ra, datos):
    """
    Predecir valores de Vc, t, y f para un valor de Ra.
    """
    # Primero buscar coincidencia exacta
    Vc_exacto, t_exacto, f_exacto = buscar_coincidencia_exacta(Ra, datos)
    
    if Vc_exacto is not None:
        return Vc_exacto, t_exacto, f_exacto
    
    # Si no hay coincidencia exacta, usar modelos de red neuronal
    modelo_Vc, modelo_t, modelo_f = modelos
    Ra_scaled = scaler.transform([[Ra]])
    
    Vc_pred = modelo_Vc.predict(Ra_scaled)[0][0]
    t_pred = modelo_t.predict(Ra_scaled)[0][0]
    f_pred = modelo_f.predict(Ra_scaled)[0][0]
    
    # Redondear los valores predichos
    Vc_pred_redondeado = round(Vc_pred)  # Redondear al entero más cercano
    Vc_pred_redondeado = float(f"{Vc_pred_redondeado}.00")  # Forzar formato .00
    t_pred_redondeado = round(t_pred, 3)  # Redondear a 3 decimales
    f_pred_redondeado = round(f_pred, 3)  # Redondear a 3 decimales
    
    return Vc_pred_redondeado, t_pred_redondeado, f_pred_redondeado

def guardar_datos(archivo, Vc, t, f, Ra):
    """
    Guardar datos predichos en el archivo CSV.
    """
    try:
        # Leer archivo existente
        datos_existentes = pd.read_csv(archivo, delimiter='\t')
        
        # Verificar si ya existe una entrada con estos valores
        duplicado = datos_existentes[
            (np.abs(datos_existentes['Vc'] - Vc) < 1e-3) & 
            (np.abs(datos_existentes['t'] - t) < 1e-3) & 
            (np.abs(datos_existentes['f'] - f) < 1e-3) & 
            (np.abs(datos_existentes['Ra'] - Ra) < 1e-3)
        ]
        
        if not duplicado.empty:
            print("Los valores ya existen en el archivo.")
            return
        
        # Preguntar detalles antes de guardar
        print("\nDetalles a guardar:")
        print(f"Velocidad de corte (Vc): {Vc:.2f} m/min")
        print(f"Profundidad de corte (t): {t:.3f} mm")
        print(f"Avance de corte (f): {f:.3f} mm/rev")
        print(f"Rugosidad aparente (Ra): {Ra:.3f} µm")
        
        confirmar = input("¿Está seguro de guardar estos valores? (s/n): ").strip().lower()
        
        if confirmar == 's':
            # Agregar nueva fila
            nueva_fila = pd.DataFrame({
                'Vc': [Vc],
                't': [t],
                'f': [f],
                'Ra': [Ra]
            })
            datos_completos = pd.concat([datos_existentes, nueva_fila], ignore_index=True)
            
            # Guardar archivo
            datos_completos.to_csv(archivo, sep='\t', index=False)
            print("Los valores se guardaron exitosamente en el archivo.")
        else:
            print("Guardado cancelado.")
    
    except Exception as e:
        print(f"Error al guardar los datos: {e}")

def main():
    """
    Función principal para predecir parámetros de corte.
    """
    archivo = 'DATOS RUGOSIDAD 4.csv'
    print("Cargando datos...")
    datos = cargar_datos(archivo)
    
    if datos is None or datos.empty:
        print("No se pudo cargar ningún dato. Cerrando programa.")
        return
    
    # Entrenar modelos
    print("Entrenando modelos de red neuronal...")
    modelos = entrenar_modelos_keras(datos)
    modelo_Vc, modelo_t, modelo_f, scaler = modelos
    
    while True:
        try:
            print("\n=== Predicción de Parámetros de Corte ===")
            Ra = float(input("Ingrese la rugosidad aparente (Ra) en µm: "))
            
            if Ra <= 0:
                print("El valor de Ra debe ser positivo.")
                continue
            
            # Predecir valores
            Vc, t, f = predecir_valores(modelos[:3], scaler, Ra, datos)
            
            print("\nParámetros de corte predichos:")
            print(f"- Velocidad de corte (Vc): {Vc:.2f} m/min")
            print(f"- Profundidad de corte (t): {t:.3f} mm")
            print(f"- Avance de corte (f): {f:.3f} mm/rev")
            
            # Preguntar si desea guardar
            guardar = input("¿Desea guardar estos valores? (s/n): ").strip().lower()
            if guardar == 's':
                guardar_datos(archivo, Vc, t, f, Ra)
            
            continuar = input("\n¿Desea hacer otra predicción? (s/n): ").strip().lower()
            if continuar != 's':
                break
        
        except ValueError:
            print("Por favor ingrese un valor numérico válido para Ra.")
        except Exception as e:
            print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()   