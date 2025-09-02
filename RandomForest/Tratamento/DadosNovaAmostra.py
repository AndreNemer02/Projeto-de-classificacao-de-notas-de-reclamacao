import pandas as pd
import re
import unidecode
import nltk
from nltk.corpus import stopwords

# Função para limpar o texto
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Retorna uma string vazia caso o valor não seja uma string
    
    # Inserir espaço após dois pontos se não houver espaço
    text = re.sub(r':(?!\s)', ': ', text)

    # Remover números, mas manter espaços
    text = re.sub(r'\d+', '', text)  # Remove todos os números

    # Remover caracteres não alfabéticos (exceto espaços)
    text = re.sub(r'[^a-zA-ZáéíóúãõçâêôàèùÁÉÍÓÚÃÕÇÂÊÔÀÈÙ\s]', '', text)
    
    # Remover acentos (com unidecode)
    text = unidecode.unidecode(text)
    
    # Converte para minúsculas
    text = text.lower()

    # Remove múltiplos espaços consecutivos
    text = re.sub(r'\s+', ' ', text).strip()

    # Substituir todas as ocorrências de 'xd' por um espaço
    text = re.sub(r'xd', ' ', text)

    text = re.sub(r'\bn\b', '', text)


    return text

nltk.download('stopwords')

# Lista de stopwords em português
stopwords_pt = set(stopwords.words('portuguese'))

def remove_stopwords(text):
    if not isinstance(text, str):
        return ""
    palavras = text.split()
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stopwords_pt]
    return ' '.join(palavras_filtradas)

# Função para substituir valores na coluna "Email/SMS/Carta"
def substituir_canal_resposta(value):
    value = str(value).strip()
    
    # Verificar primeiro se é um e-mail
    if re.match(r'^[^@]+@[^@]+\.[^@]+$', value):  # E-mail
        return 1
    
    # Verificar se é um número de telefone
    if re.match(r'^[\d\(\)\-\s]{9,15}$', value):  # Número de telefone
        clean_value = re.sub(r'\D', '', value)  # Remover caracteres não numéricos
        if len(clean_value) >= 9:
            return 0
    
    # Verificar padrões específicos
    patterns = {
        'PRO': 1,
        r'.*BR$': 3,
        'improcedente': 1,
        'procedente': 1,
        'Canais de atendimento': 2,
        'Envio manual DECT': 1,
        'ENVIADA PELA DECG - em anexo': 1,
        'sem contato': 9,
        'CANAL DE ATENDIMENTO': 2
    }
    for pattern, replacement in patterns.items():
        if re.match(pattern, value, re.IGNORECASE):
            return replacement
    
    # Verificar frases relacionadas a e-mail ou anexos
    if any(phrase in value.lower() for phrase in ['em anexo', 'vide anexo', 'e-mail', 'enviado e-mail']):
        return 1
    
    # Valor padrão caso nenhuma regra se aplique
    return value

# Carregar a planilha existente
planilha = pd.read_excel('NovaAmostra.xlsx', sheet_name=None)
nome_planilha = list(planilha.keys())[0]
planilha = planilha[nome_planilha]

# Garantir que todos os valores nulos sejam substituídos por 9
planilha['Email/SMS/Carta'] = planilha['Email/SMS/Carta'].fillna(9).astype(str)

# Substituir valores usando a função
planilha['Email/SMS/Carta'] = planilha['Email/SMS/Carta'].apply(substituir_canal_resposta)

# Substituir valores nas outras colunas
planilha['Status'] = planilha['Status'].replace({'ENCI': 0,'ENCP': 1,'VIMP': 0,'VPRO': 1,'improcedente': 0,'procedente': 1,'#N/D': 2,'VERI': 0})
planilha['Tipo de Resposta'] = planilha['Tipo de Resposta'].replace({'-': 9, 'SMS': 0, 'E-MAIL': 1, 'CANAL DE ATENDIMENTO': 2, 'CARTA': 3})

# Remover duplicatas
planilha = planilha.drop_duplicates()

# Substituir valores vazios ou nulos nas colunas restantes por 9
planilha = planilha.fillna(9).replace('', 9)

# Garantir que o tipo de dados dos objetos seja tratado corretamente
planilha = planilha.infer_objects()

# Aplicar a limpeza de texto e a remoção de stopwords
planilha['Descricao limpa'] = planilha['Descrição'].apply(clean_text).apply(remove_stopwords)
planilha['Resposta limpa'] = planilha['Resposta'].apply(clean_text).apply(remove_stopwords)

planilha.to_excel('NovaAmostraTratadaRF.xlsx', index=False)

print("Pré-processamento concluído e planilha salva com sucesso.")

