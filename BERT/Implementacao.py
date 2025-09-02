import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from torch.nn.functional import softmax
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings['input_ids'])

def tokenize_texts(tokenizer, texts, max_length=512):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def aplicar_regras_tipo2(df):
    df['Tipo de Resposta'] = df['Tipo de Resposta'].astype(str)
    df['Email/SMS/Carta'] = df['Email/SMS/Carta'].astype(str)
    df['Descrição'] = df['Descrição'].astype(str)
    df['Resposta'] = df['Resposta'].astype(str)

    cond1 = (
        ((df['Tipo de Resposta'] != df['Email/SMS/Carta']) |
         ((df['Tipo de Resposta'] == '9') & (df['Email/SMS/Carta'] == '9')))
        & ~((df['Tipo de Resposta'] == '2') & (df['Email/SMS/Carta'] == '9'))
    )

    cond2 = (
        (df['Descrição'].str.strip().isin(['9', '-'])) |
        (df['Resposta'].str.strip().isin(['9', '-']))
    )

    df['Nao_Conforme_Regras_Tipo2'] = (cond1 | cond2).astype(int)
    return df

def main():
    save_path = "Coloque o local no qual o modelo BERT está salvo"
    train_file = "DadosParaTreino.xlsx"
    test_file = "NovaAmostraTratada.xlsx"
    threshold = 0.50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)

    def concat_texts(df):
        return (df['Status'].astype(str).fillna('') + ' ' +
                df['Descrição'].astype(str).fillna('') + ' ' +
                df['Resposta'].astype(str).fillna('')).tolist()

    train_texts = concat_texts(train_df)
    test_texts = concat_texts(test_df)
    train_labels = train_df['Em conformidade'].tolist()

    ros = RandomOverSampler(random_state=42)
    train_texts, train_labels = ros.fit_resample(np.array(train_texts).reshape(-1, 1), train_labels)
    train_texts = train_texts.flatten().tolist()

    print("Distribuição das classes:", Counter(train_labels))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = BertForSequenceClassification.from_pretrained(save_path, num_labels=2).to(device)

    train_encodings = tokenize_texts(tokenizer, train_texts)
    val_encodings = tokenize_texts(tokenizer, val_texts)
    test_encodings = tokenize_texts(tokenizer, test_texts)

    train_dataset = CustomDataset(train_encodings, train_labels)
    val_dataset = CustomDataset(val_encodings, val_labels)
    test_dataset = CustomDataset(test_encodings)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=6,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("\nMétricas de avaliação na validação:", eval_results)

    model.save_pretrained(save_path, safe_serialization=False)
    tokenizer.save_pretrained(save_path)
    print(f"Modelo e tokenizer salvos em: {save_path}")

    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    predictions_bert, probs_bert = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = softmax(outputs.logits, dim=-1)
            preds = (probs[:, 1] >= threshold).long().cpu().tolist()
            predictions_bert.extend(preds)
            probs_bert.extend(probs[:, 1].cpu().numpy())

    test_df['Texto'] = test_texts
    test_df['Em_Conformidade_BERT'] = predictions_bert
    test_df['Probabilidade_Conforme'] = probs_bert
    test_df = aplicar_regras_tipo2(test_df)

    if 'Em conformidade' in test_df.columns:
        y_true = test_df['Em conformidade'].values
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, predictions_bert))
        print(f"Precision: {precision_score(y_true, predictions_bert, average='macro'):.4f}")
        print(f"Recall: {recall_score(y_true, predictions_bert, average='macro'):.4f}")
        print(f"F1-score: {f1_score(y_true, predictions_bert, average='macro'):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_true, probs_bert):.4f}")
        print(f"Log-Loss: {log_loss(y_true, probs_bert):.4f}")

    output_path = os.path.join(os.path.dirname(test_file), 'analise_completa_conformidade.xlsx')
    test_df.to_excel(output_path, index=False)
    print(f"Análise completa salva em: {output_path}")

if __name__ == '__main__':
    main()
