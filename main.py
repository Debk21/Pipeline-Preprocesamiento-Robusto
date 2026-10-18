#importacion de librerias

import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Carga de dataset
datos = fetch_california_housing(as_frame=True)

#Separacion de variables
X = datos.data.copy()
y = datos.target.copy()

#Creacion de variable categorica a partir del ingreso medio
X["categoria_ingreso"] = pd.cut(
    X["MedInc"],
    bins=[-float("inf"), 2, 4, float("inf")],
    labels=["Bajo", "Medio", "Alto"]
).astype("object")

#Conocimiento de datos
print("Primeras filas del dataset:")
print(X.head())

#Consulta de cantidad de filas y columnas
print("\nDimensiones del dataset:")
print(X.shape)

#Valores faltantes
print("\nValores faltantes por columna:")
print(X.isnull().sum())

#Division de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Comprobacion de tamaño de cada conjunto
print("\nFilas para entrenamiento:", len(X_train))
print("Filas de prueba:", len(X_test))

#Identificacion de tipo de variables
columnas_numericas = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

columnas_categoricas = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nColumnas numericas:", columnas_numericas)
print("Columnas categoricas:", columnas_categoricas)

#Preprocesamiento de variables numericas
pipeline_numerico = Pipeline(steps=[
    ("imputacion", SimpleImputer(strategy="median")),
    ("escalado", StandardScaler())
])

#Preprocesamiento de variables categoricas
pipeline_categorico = Pipeline(steps=[
    ("imputacion", SimpleImputer(strategy="most_frequent")),
    ("codificacion", OneHotEncoder(handle_unknown="ignore"))
])

#Integracion del preprocesamiento
preprocesador = ColumnTransformer(transformers=[
    ("numericas", pipeline_numerico, columnas_numericas),
    ("categoricas", pipeline_categorico, columnas_categoricas)
])

#Construccion del Pipeline final
pipeline_final = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("modelo", LinearRegression())
])

#Entrenamiento del pipeline
pipeline_final.fit(X_train, y_train)

print("\nPipeline entrenado correctamente.")

#Evaluacion del modelo con los datos de prueba
predicciones = pipeline_final.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
rmse = mse ** 0.5
r2 = r2_score(y_test, predicciones)

#Metricas obtenidas
print("\n--- Metricas del modelo ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

#Prueba con un registro creado manualmente
nuevo_registro = pd.DataFrame([{
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 5.5,
    "AveBedrms": 1.1,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0,
    "categoria_ingreso": "Alto"
}])

#Prediccion con registro nuevo
predicciones_nueva = pipeline_final.predict(nuevo_registro)

print("\n--- Prediccion para un registro nuevo ---")
print(f"Valor estimado: {predicciones_nueva[0]: .4f}")


#Guardar pipeline entrenado
joblib.dump(pipeline_final, "pipeline_viviendas.joblib")

print("\nPipeline guardado correctamente")