Predicción de reincidencia en infractores de tránsito – Arequipa 2024
Este repositorio contiene el código fuente completo del modelo predictivo desarrollado como parte de una investigaciion en Gestión Pública. El objetivo fue comparar el desempeño de los algoritmos Random Forest y Regresión Logística en la predicción de reincidencia en infracciones de tránsito, utilizando una base de datos institucional del año 2024 proveniente de la Municipalidad Provincial de Arequipa (MPA).

📌 Funcionalidades principales
Limpieza y transformación de datos reales sobre papeletas.

Ingeniería de variables como:

Reincidencia (variable objetivo)

Intervalo entre papeletas

Pronto pago

Día de la semana

Preprocesamiento con OneHotEncoder y ColumnTransformer.

Construcción de pipelines en scikit-learn.

Entrenamiento y evaluación de modelos:

Accuracy

F1-score

AUC–ROC

Matrices de confusión

Curvas ROC

Visualización comparativa de métricas.

Identificación de las variables más influyentes según importancia (Random Forest) y coeficientes (Regresión Logística).

🗂 Estructura del código
Carga y depuración de datos

Creación de nuevas variables con base en fechas y códigos de infracción

Construcción de modelos predictivos

Evaluación de resultados con validación cruzada

Visualización de resultados

⚙️ Requisitos
Python 3.9+

pandas

scikit-learn

matplotlib

seaborn

openpyxl

data set : https://www.datosabiertos.gob.pe/dataset/listado-de-papeletas-enero-2024-diciembre-2024-municipalidad-provincial-de-arequipa-mpa
