
# Pipeline de Preprocesamiento Robusto

## Descripción del proyecto

Este proyecto consiste en construir un pipeline de preprocesamiento de datos utilizando Python y Scikit-learn.

Se utiliza el dataset California Housing para preparar los datos, transformar variables numéricas y categóricas, entrenar un modelo de regresión y evaluar sus resultados.

El objetivo es integrar las etapas de preprocesamiento y modelado en un único pipeline reproducible.


## Herramientas utilizadas

- Python
- Pandas: manipulación y análisis de datos.
- Scikit-learn: preprocesamiento, entrenamiento y evaluación del modelo.
- Joblib: almacenamiento del pipeline entrenado.

## Dataset

Se utiliza California Housing, un conjunto de datos que contiene información sobre viviendas y características de distintas zonas de California.

La variable objetivo es el valor medio de las viviendas. Se utiliza un modelo de regresión lineal para realizar las predicciones.


## Cómo ejecutar el proyecto

Para ejecutar el proyecto, es necesario tener Python instalado y las librerías que se utilizaron.

Primero, se deben instalar las dependencias. Después, se ejecuta el archivo `main.py`, que carga los datos, realiza el preprocesamiento, entrena el modelo y muestra las métricas obtenidas.

Al finalizar, se guarda el pipeline entrenado en un archivo llamado `pipeline_viviendas.joblib`.


## Resultados

El dataset utilizado contiene 20.640 registros y 9 variables, incluyendo la variable categórica que se agregó durante el análisis.

Se dividieron los datos en un conjunto de entrenamiento y otro de prueba. Luego, se entrenó el pipeline con un modelo de regresión lineal.

Las métricas obtenidas fueron:

- MAE: 0.5342
- RMSE: 0.7458
- R²: 0.5755

Con este trabajo pude practicar cómo combinar el preprocesamiento de variables numéricas y categóricas con el entrenamiento de un modelo, utilizando un pipeline de Scikit-learn.
