Informe Detallado 🌟

Introducción 📚

En este proyecto se desarrolló un sistema basado en Python que emplea técnicas de aprendizaje automático y redes neuronales para predecir parámetros de corte en función de la rugosidad superficial (Ra). La implementación incluye:

Preprocesamiento de datos.

Entrenamiento de modelos de redes neuronales.

Predicción y almacenamiento de resultados.

Objetivo 🎯

El principal objetivo fue diseñar una solución automatizada para predecir los parámetros de corte (Vc, t, f) a partir de la rugosidad (Ra) de una muestra, optimizando la precisión de las predicciones y garantizando una interfaz amigable.

Desarrollo del Sistema ⚙️

1. Carga de Datos 📥

Se creó la función cargar_datos, que permite:

Leer archivos CSV con múltiples intentos de delimitadores (\t, ;, ,,  ).

Verificar y mapear columnas requeridas: Vc, t, f, Ra.

Eliminar valores no numéricos o faltantes para garantizar datos limpios.

2. Preparación de Datos 🧪

Mediante preparar_datos, los datos se dividieron en conjuntos de entrenamiento y prueba:

Uso de train_test_split para separar datos.

Escalado de características usando StandardScaler.

3. Creación y Entrenamiento de Modelos 🤖

Se implementaron redes neuronales mediante TensorFlow:

Arquitectura:

Capa de entrada: 1 neurona.

Capas ocultas: 128, 64 y 32 neuronas con activación ReLU.

Capa de salida: 1 neurona.

Optimizador: Adam.

Función de pérdida: MSE.

Early Stopping para evitar sobreajuste.

4. Predicción de Valores 🔮

La función predecir_valores permite:

Buscar coincidencias exactas en los datos existentes.

Predecir valores usando modelos entrenados.

Redondear los resultados para mayor claridad.

5. Almacenamiento de Resultados 💾

Se diseñó guardar_datos para almacenar valores predichos en un archivo CSV, verificando duplicados antes de añadir nuevas filas.

Resultados Obtenidos 📊

Precisión de los Modelos:

Modelo Vc: pérdida mínima en validación.

Modelo t: pérdida mínima en validación.

Modelo f: pérdida mínima en validación.

Ejemplo de Predicción:

Entrada (Ra): 2.500 µm.

Salida:

Vc: 300.00 m/min.

t: 0.250 mm.

f: 0.200 mm/rev.

Conclusiones ✅

El sistema automatizado garantiza precisión y flexibilidad en la predicción de parámetros de corte.

La implementación modular facilita futuras mejoras y adaptaciones.

La interfaz basada en consola es intuitiva para usuarios con conocimientos básicos de Python.

Recomendaciones 🚀

Integrar una interfaz gráfica (GUI) para mejorar la experiencia del usuario.

Ampliar el conjunto de datos para entrenar modelos más robustos.

Implementar soporte para más formatos de archivo.

Agradecimientos 🙌

Gracias a las bibliotecas Pandas, NumPy, TensorFlow y Scikit-learn por simplificar la implementación de este proyecto.

![image](https://github.com/user-attachments/assets/6fcde41e-1f8f-49d6-8eba-cfec99180110)
![image](https://github.com/user-attachments/assets/e4928543-f13c-4f82-8d06-563c01eccbdd)
![image](https://github.com/user-attachments/assets/29609929-167b-4e42-abd4-663115f6b894)

📝 Contactos Whatsapp: (+51 928221439) https://wa.link/237bkx Correo: tecnologiasandoval2023@gmail.com Github @arturosandoval19 ·
Youtube:https://www.youtube.com/@tecnolog%C3%ADaSandoval Facebook: https://www.facebook.com/tecnologiasandoval1 TikTok: www.tiktok.com/@tecnologiasandoval


