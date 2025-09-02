import pandas as pd
import re
import unidecode

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


# Função para tratar as colunas substituindo valores e preenchendo nulos
def tratar_coluna(coluna, substituicoes, valor_nulo=9):
    planilha[coluna] = planilha[coluna].replace(substituicoes).fillna(valor_nulo)

# Carregar a planilha existente
planilha = pd.read_excel('NovaAmostra.xlsx', sheet_name=None)
nome_planilha = list(planilha.keys())[0]
planilha = planilha[nome_planilha]

# Garantir que todos os valores nulos sejam substituídos por 9
planilha['Email/SMS/Carta'] = planilha['Email/SMS/Carta'].fillna(9).astype(str)

# Substituir valores usando a função
planilha['Email/SMS/Carta'] = planilha['Email/SMS/Carta'].apply(substituir_canal_resposta)

# Substituir valores na coluna 'Status' com explicações mais claras
substituicoes_status = {
    'ENCI': 'Improcedente.',
    'ENCP': 'Procedente.',
    'VIMP': 'Improcedente.',
    'VPRO': 'Procedente.',
    'improcedente': 'Improcedente.',
    'procedente': 'Procedente.',
    '#N/D': 2,
    'VERI': 'Improcedente.'
}
tratar_coluna('Status', substituicoes_status)

# Substituir valores nas outras colunas usando a função 'tratar_coluna'
tratar_coluna('Tipo de Resposta', {'-': 9, 'SMS': 0, 'E-MAIL': 1, 'CANAL DE ATENDIMENTO': 2, 'CARTA': 3})

# Substituir valores vazios ou nulos nas colunas restantes por 9
planilha = planilha.fillna(9).replace('', 9)

# Garantir que o tipo de dados dos objetos seja tratado corretamente
planilha = planilha.infer_objects()

# Salvar o resultado em um novo arquivo Excel
planilha.to_excel('NovaAmostraTratada.xlsx', index=False)

print("Pré-processamento concluído e planilha salva com sucesso.")
