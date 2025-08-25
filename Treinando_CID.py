# Modelo que melhor se aproximou do esperado
# Treinamento de um modelo spaCy para reconhecimento de entidades relacionadas a CID
# Baseado no arquivo Treinando_CID.py

import os
import spacy
from spacy.tokens import DocBin
import json
import subprocess
from pathlib import Path
import re
import random

# Função para extrair texto de forma segura
def extrair_texto(item):
    texto = item[0]
    
    if isinstance(texto, str):
        return texto
    
    if isinstance(texto, list):
        partes_texto = []
        for elemento in texto:
            if isinstance(elemento, str):
                partes_texto.append(elemento)
            elif isinstance(elemento, dict) and 'text' in elemento:
                partes_texto.append(elemento['text'])
            else:
                partes_texto.append(str(elemento))
        return " ".join(partes_texto)
    
    return str(texto)

# Função para ajustar anotações de CID
def ajustar_anotacoes_cid(texto, entities):
    # padrao_cid = re.compile(r'(CID[:\-]?\s*([A-Z]\d+(?:\.\d+)?))', re.IGNORECASE) # Modelo funcional
    # Ajuste para capturar CID com ou sem espaços e com diferentes formatações
    padrao_cid = re.compile(r'(CID[:\-]?\s*([A-Z][\d_.-]*))', re.IGNORECASE) # Testando uma nova regex
    novas_entidades = []
    
    for match in padrao_cid.finditer(texto):
        texto_completo, codigo_cid = match.groups()
        start = match.start()
        end = match.end()
        
        start_codigo = start + len(texto_completo) - len(codigo_cid)
        novas_entidades.append([start_codigo, start_codigo + len(codigo_cid), "CID"])
    
    return novas_entidades

# Função para gerar dados sintéticos
def gerar_dados_sinteticos():
    doencas = [
        ("gripe", "J11.1"),
        ("depressão", "F33.9"),
        ("diabetes", "E11.9"),
        ("hipertensão", "I10"),
        ("fratura", "S02.5"),
        ("asma", "J45.909"),
        ("gastrite", "K29.0"),
        ("artrite", "M06.9"),
        ("conjuntivite", "H10.9"),
        ("ansiedade", "F41.1")
    ]
    
    formatos = [
        "CID:{}",
        "CID {}",
        "Código CID: {}",
        "({})",
        "CID-10: {}",
        "{}",
        "(CID {})",
        "CID: {}",
        "CID-10 {}",
        "código {}"
    ]
    
    templates = [
        "Diagnóstico: {} {}",
        "Identificado {} - {}",
        "CID registrado: {} para {}",
        "Paciente com {} ({})",
        "Confirmado {} - caso de {}",
        "{} relacionado a {}",
        "CID atribuído: {} para condição de {}",
        "Código de doença: {} ({})",
        "{} correspondente a {}",
        "CID identificado: {} em caso de {}"
    ]
    
    dados_sinteticos = []
    for doenca, cid in doencas:
        for _ in range(5):
            formato = random.choice(formatos)
            template = random.choice(templates)
            
            cid_formatado = formato.format(cid)
            texto = template.format(cid_formatado, doenca)
            
            start = texto.find(cid)
            if start == -1:
                continue
                
            end = start + len(cid)
            dados_sinteticos.append((texto, {'entities': [[start, end, "CID"]]}))
    
    return dados_sinteticos

# Função para carregar e ajustar os dados
def carregar_dados(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    treinamentos = []
    padrao_cid = re.compile(r'[A-Z]\d+(\.\d+)?')
    
    for i, item in enumerate(data):
        if len(item) < 2:
            print(f"Item {i} inválido: não possui elementos suficientes")
            continue
            
        texto = extrair_texto(item)
        anotacao = item[1]
        
        entities = []
        if isinstance(anotacao, list):
            for elem in anotacao:
                if isinstance(elem, dict) and 'entities' in elem:
                    entities = elem['entities']
                    break
        elif isinstance(anotacao, dict) and 'entities' in anotacao:
            entities = anotacao['entities']
        
        entidades_ajustadas = []
        for ent in entities:
            if isinstance(ent, list) and len(ent) > 2 and ent[2] == 'CID':
                texto_entidade = texto[ent[0]:ent[1]]
                if not padrao_cid.match(texto_entidade):
                    novos_cids = ajustar_anotacoes_cid(texto, [ent])
                    if novos_cids:
                        entidades_ajustadas.extend(novos_cids)
                        continue
                entidades_ajustadas.append(ent)
        
        if entidades_ajustadas:
            treinamentos.append((texto, {'entities': entidades_ajustadas}))
    
    print(f"Total de itens processados: {len(treinamentos)}")
    if treinamentos:
        print(f"Primeiro item processado: {treinamentos[0][0][:50]}...")
        print(f"Entidades do primeiro item: {treinamentos[0][1]['entities']}")
    return treinamentos

# Carregar e aumentar dados de treino e validação
base_train = carregar_dados("ner_treino_split.json")
base_dev = carregar_dados("ner_validacao_split.json")

# Gerar dados sintéticos
dados_sinteticos = gerar_dados_sinteticos()

# Combinar dados originais com sintéticos
train_data = base_train + dados_sinteticos
dev_data = base_dev + dados_sinteticos

print(f"\nDados de treino aumentados: {len(base_train)} originais + {len(dados_sinteticos)} sintéticos = {len(train_data)} total")
print(f"Dados de validação aumentados: {len(base_dev)} originais + {len(dados_sinteticos)} sintéticos = {len(dev_data)} total")

# Inicializar pipeline com NER
nlp = spacy.blank("pt")
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Adicionar rótulo "CID"
ner.add_label("CID")

# Criar DocBin (formato otimizado para spaCy)
def criar_docbin(data):
    doc_bin = DocBin()
    for texto, anotacao in data:
        if not isinstance(texto, str):
            print(f"Texto inválido encontrado: {texto}")
            continue
            
        doc = nlp.make_doc(texto)
        ents = []
        for ent in anotacao["entities"]:
            if len(ent) < 3:
                continue
            start, end, label = ent[0], ent[1], ent[2]
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)
    return doc_bin

# Salvar os dados em formato spaCy
criar_docbin(train_data).to_disk("train_CID.spacy")
criar_docbin(dev_data).to_disk("dev_CID.spacy")

# Obter o diretório atual do script
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Criar caminho para salvar o modelo
caminho_modelo = os.path.join(diretorio_atual, "modelo_NER_CID")

# Gera o arquivo de configuração
subprocess.run([
    "spacy", "init", "config",
    "--lang", "pt",
    "--pipeline", "ner",
    "--force",
    "config.cfg"
])

# Treinar o modelo
subprocess.run([
    "spacy", "train",
    "config.cfg",
    "--output", caminho_modelo,
    "--paths.train", "train_CID.spacy",
    "--paths.dev", "dev_CID.spacy"
])

print(f"Modelo treinado salvo em: {caminho_modelo}")