"""
Created on Tue May 20 13:58:38 2025

@author: leger Hardy Fernandez Meza
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Cargar datos
df = pd.read_excel(
    r"C:\Users\leger4\Downloads\MODELO MACHINE LEARNING TESIS\Papeletas_2024-MPA.xlsx",
    sheet_name="_Select_PAP_PAP_insoluto_PAP_ad"
)

# Limpieza y preparación inicial
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['fecha_papeleta'] = pd.to_datetime(df['fecha_papeleta'], errors='coerce')
df['fecha_de_pago'] = pd.to_datetime(df['fecha_de_pago'], errors='coerce')

# Lista de infracciones sin derecho a descuento
sin_descuento = {"M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M12",
                 "M16", "M17", "M20", "M21", "M23", "M27", "M28", "M29", "M31", "M32", "M42"}

# Nueva variable: fue_pagada
df['fue_pagada'] = (df['estado_papeleta'].str.lower() == 'pagada').astype(int)

# Ordenar para calcular intervalos entre papeletas
df = df.sort_values(by=['identificador_persona', 'fecha_papeleta'])
df['dias_desde_anterior'] = df.groupby('identificador_persona')['fecha_papeleta'].diff().dt.days

# Categorización del intervalo
def categorizar_intervalo(dias):
    if pd.isna(dias):
        return 'Primera'
    elif dias <= 1:
        return 'Inmediata'
    elif dias <= 7:
        return 'Corta'
    elif dias <= 30:
        return 'Media'
    else:
        return 'Larga'

df['intervalo_entre_papeletas'] = df['dias_desde_anterior'].apply(categorizar_intervalo)

# Variable pronto_pago
df['dias_para_pago'] = (df['fecha_de_pago'] - df['fecha_papeleta']).dt.days
df['pronto_pago'] = (
    (df['fue_pagada'] == 1) &
    (~df['codigo_infraccion'].str.upper().isin(sin_descuento)) &
    (df['dias_para_pago'] <= 8)
).astype(int)

# Día de la semana
df['dia_semana'] = df['fecha_papeleta'].dt.day_name()

# Variable objetivo: reincidencia general
conteo = df['identificador_persona'].value_counts()
df['reincidencia'] = df['identificador_persona'].map(lambda x: 1 if conteo[x] > 1 else 0)

# Variables para el modelo
features = ['codigo_infraccion', 'estado_papeleta', 'dia_semana', 'pronto_pago',
            'intervalo_entre_papeletas', 'fue_pagada']
target = 'reincidencia'
df_modelo = df[features + [target]].dropna()

# Separar datos
X = df_modelo[features]
y = df_modelo[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)

# Preprocesamiento (solo categóricas)
categorical_features = ['codigo_infraccion', 'estado_papeleta', 'dia_semana', 'intervalo_entre_papeletas']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Pipelines
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=40))
])
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=40))
])

# Entrenamiento
pipeline_rf.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)

# Predicciones
y_pred_rf = pipeline_rf.predict(X_test)
y_prob_rf = pipeline_rf.predict_proba(X_test)[:, 1]
y_pred_lr = pipeline_lr.predict(X_test)
y_prob_lr = pipeline_lr.predict_proba(X_test)[:, 1]

# Reportes
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
lr_report = classification_report(y_test, y_pred_lr, output_dict=True)

comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'F1-score', 'Recall', 'Precision', 'AUC-ROC'],
    'Random Forest': [
        rf_report['accuracy'],
        rf_report['1']['f1-score'],
        rf_report['1']['recall'],
        rf_report['1']['precision'],
        roc_auc_score(y_test, y_prob_rf)
    ],
    'Regresión Logística': [
        lr_report['accuracy'],
        lr_report['1']['f1-score'],
        lr_report['1']['recall'],
        lr_report['1']['precision'],
        roc_auc_score(y_test, y_prob_lr)
    ]
})

print(comparison_df)

# MATRICES DE CONFUSIÓN
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf),
                       display_labels=['No Reincidente', 'Reincidente']).plot(cmap='Blues', ax=plt.gca())
plt.title("Matriz de Confusión - Random Forest")

plt.subplot(1, 2, 2)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr),
                       display_labels=['No Reincidente', 'Reincidente']).plot(cmap='Oranges', ax=plt.gca())
plt.title("Matriz de Confusión - Regresión Logística")
plt.tight_layout()
plt.show()

# CURVA ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.3f})')
plt.plot(fpr_lr, tpr_lr, label=f'Regresión Logística (AUC = {roc_auc_score(y_test, y_prob_lr):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC Comparativa')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# COMPARACIÓN DE MÉTRICAS
comparison_melted = comparison_df.melt(id_vars='Métrica', var_name='Modelo', value_name='Valor')
plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_melted, x='Métrica', y='Valor', hue='Modelo')
plt.title('Comparación de Métricas entre Modelos (Mixto: pagadas y no pagadas)')
plt.ylabel('Valor')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Al finalizar el análisis de modelos, se puede observar lo siguiente:
# Aquí tienes el Top 10 de variables más importantes para cada modelo:
# En Random Forest, las variables aparecen ordenadas según su contribución a la reducción del error.
# En Regresión Logística, los coeficientes están ordenados por magnitud absoluta, indicando fuerza y dirección del efecto.

# Código para mostrar top 10 de cada modelo:
feature_names = pipeline_rf.named_steps['preprocessor'].get_feature_names_out()
rf_importances = pipeline_rf.named_steps['classifier'].feature_importances_
lr_coef = pipeline_lr.named_steps['classifier'].coef_[0]

rf_importance_df = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': rf_importances
}).sort_values(by='Importancia', ascending=False)

lr_importance_df = pd.DataFrame({
    'Variable': feature_names,
    'Coeficiente': lr_coef
}).sort_values(by='Coeficiente', key=abs, ascending=False)

top10_rf = rf_importance_df.head(10)
top10_lr = lr_importance_df.head(10)

print("Top 10 - Random Forest:")
print(top10_rf)

print("\nTop 10 - Regresión Logística:")
print(top10_lr)