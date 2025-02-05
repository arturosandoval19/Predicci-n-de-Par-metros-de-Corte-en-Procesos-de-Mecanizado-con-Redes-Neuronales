Informe: Predicción de Parámetros de Corte Basada en Rugosidad

Introducción

Este proyecto tiene como objetivo predecir parámetros de corte —velocidad de corte (Vc), profundidad de corte (t) y avance de corte (f)— a partir de la rugosidad aparente (Ra) utilizando un enfoque basado en redes neuronales. El sistema procesa datos de un archivo CSV, los prepara para entrenamiento, construye modelos predictivos y permite realizar predicciones con posibilidad de guardar los resultados.

Desarrollo

1. Carga de Datos

Se implementó una función llamada cargar_datos para leer un archivo CSV con diferentes delimitadores (",", ";", "\t", y espacios) con el objetivo de maximizar la compatibilidad con diversos formatos. La función realiza las siguientes tareas:

Detecta y asigna columnas necesarias: Vc, t, f, y Ra.

Renombra columnas automáticamente en caso de discrepancias.

Convierte datos a formato numérico, eliminando filas con valores faltantes.

2. Búsqueda de Coincidencias Exactas

Se diseñó la función buscar_coincidencia_exacta, que permite identificar registros existentes en el archivo cuya rugosidad (Ra) coincida con un valor dado dentro de un margen de tolerancia. Esto optimiza el uso de datos existentes sin necesidad de realizar predicciones adicionales.

3. Preparación de Datos para Entrenamiento

La función preparar_datos divide el conjunto de datos en subconjuntos de entrenamiento y prueba, separando las características (Ra) de los objetivos (Vc, t, f). Además, escala las características para mejorar el rendimiento del modelo utilizando StandardScaler.

4. Creación y Entrenamiento de Modelos de Redes Neuronales

Se definió la arquitectura de una red neuronal en la función crear_modelo_keras. Cada modelo tiene:

Tres capas ocultas (128, 64 y 32 neuronas) con activación ReLU.

Una capa de salida para predecir un objetivo específico.

Los modelos predictivos para Vc, t, y f se entrenaron simultáneamente mediante la función entrenar_modelos_keras, que utiliza EarlyStopping para prevenir el sobreajuste. Cada modelo fue evaluado en el conjunto de prueba para garantizar su rendimiento.

5. Predicción de Valores

La función predecir_valores calcula los parámetros de corte:

Busca coincidencias exactas en los datos existentes.

En caso de no encontrarlas, utiliza los modelos entrenados para predecir los valores a partir de la rugosidad escalada.

Los resultados se redondean para cumplir con especificaciones de formato.

6. Almacenamiento de Resultados

La función guardar_datos verifica si los valores predichos ya existen en el archivo antes de agregarlos. Si no hay duplicados, los datos se almacenan en formato CSV para facilitar su reutilización.

7. Interfaz Principal

El programa principal (él éxito se controla desde main) realiza las siguientes acciones:

Carga los datos iniciales.

Entrena los modelos de redes neuronales.

Solicita al usuario ingresar un valor de rugosidad (Ra).

Predice los parámetros de corte y pregunta si desea guardarlos.

Permite realizar predicciones adicionales o finalizar el programa.

Resultados

Se logró una carga eficiente de datos desde archivos CSV con diferentes configuraciones.

Los modelos de redes neuronales mostraron un buen rendimiento, con errores mínimos en el conjunto de prueba.

El sistema predijo de manera precisa los parámetros de corte basándose en la rugosidad ingresada por el usuario.

Los datos generados se almacenaron con éxito, evitando duplicados.

Conclusiones

El sistema desarrollado demuestra ser una herramienta robusta y flexible para predecir parámetros de corte en base a rugosidad aparente. Las siguientes mejoras podrían optimizar aún más su funcionalidad:

Incorporar visualizaciones gráficas de las predicciones y los datos.

Implementar una interfaz gráfica para facilitar su uso.

Explorar arquitecturas de redes neuronales más avanzadas para mejorar la precisión.


![image](https://github.com/user-attachments/assets/6fcde41e-1f8f-49d6-8eba-cfec99180110)
![image](https://github.com/user-attachments/assets/e4928543-f13c-4f82-8d06-563c01eccbdd)
![image](https://github.com/user-attachments/assets/29609929-167b-4e42-abd4-663115f6b894)


