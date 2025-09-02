import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report, precision_score,
                             recall_score, f1_score, roc_auc_score, log_loss,
                             confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar dados
data = pd.read_excel('DadosParaTreinoRF.xlsx').dropna()

# 2. Nova coluna combinada de texto
data['texto_combinado'] = data['Descricao limpa'].astype(str) + " " + data['Resposta limpa'].astype(str)

X = data[['texto_combinado', 'Status']]
y = data['Em conformidade']

# 3. Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pré-processamento com ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(min_df=3, max_features=5000), 'texto_combinado'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Status'])
])

# 5. Pipeline com SMOTE + modelo
pipeline = ImbPipeline(steps=[
    ('preprocessing', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. GridSearchCV para ajustar hiperparâmetros
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_

# 7. Avaliação
y_pred = best_pipeline.predict(X_test)
y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\nResultados com Pipeline + SMOTE:")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 8. Métricas detalhadas
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
log_loss_value = log_loss(y_test, y_pred_prob)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Log-Loss: {log_loss_value:.4f}")

# 9. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# 10. Análise de threshold
thresholds = np.linspace(0, 1, 101)
precisions, recalls = [], []

for threshold in thresholds:
    y_pred_adj = (y_pred_prob >= threshold).astype("int32")
    precisions.append(precision_score(y_test, y_pred_adj, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_adj))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.title("Precision vs Recall")
plt.show()

# 11. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.legend()
plt.grid()
plt.show()

# 12. Validação cruzada
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Acurácia média (CV): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# 13. Salvar pipeline completo
joblib.dump(best_pipeline, 'modelo_pipeline_rf.pkl')

# 14. Previsão em nova amostra
def preprocess_and_predict(file_path, threshold_percentile=20):
    new_data = pd.read_excel(file_path).dropna()
    
    required_cols = ['Descricao limpa', 'Resposta limpa', 'Status', 'Tipo de Resposta', 'Email/SMS/Carta']
    if not all(col in new_data.columns for col in required_cols):
        raise ValueError(f"Arquivo deve conter as colunas: {required_cols}")

    # Regra de negócio: marcar diretamente como NÃO conforme (0)
    regra_mask = (
        ((new_data['Tipo de Resposta'] == 9) & (new_data['Email/SMS/Carta'] == 9)) |
        (new_data['Tipo de Resposta'] != new_data['Email/SMS/Carta'])
    )

    new_data['Em conformidade'] = np.nan  # Inicializa a coluna
    
    # Aplica a regra diretamente
    new_data.loc[regra_mask, 'Em conformidade'] = 0

    # Apenas as linhas que NÃO se enquadram na regra passam pelo modelo
    dados_para_modelo = new_data.loc[~regra_mask].copy()
    dados_para_modelo['texto_combinado'] = (
        dados_para_modelo['Descricao limpa'].astype(str) + " " + dados_para_modelo['Resposta limpa'].astype(str)
    )
    
    X_new = dados_para_modelo[['texto_combinado', 'Status']]
    prob = best_pipeline.predict_proba(X_new)[:, 1]
    threshold = np.percentile(prob, threshold_percentile)
    predictions = (prob >= threshold).astype("int32")

    # Atualiza os resultados no DataFrame original
    new_data.loc[~regra_mask, 'Em conformidade'] = predictions

    # Exporta os resultados
    new_data.to_excel("DadosAnalisadosPorRandomForest.xlsx", index=False)

    print(f"Limiar ajustado: {threshold:.4f}")
    print(f"Proporção de não conformidades previstas: {1 - (new_data['Em conformidade'] == 1).mean():.2%}")


# Executar em nova amostra
preprocess_and_predict("NovaAmostraTratadaRF.xlsx")
