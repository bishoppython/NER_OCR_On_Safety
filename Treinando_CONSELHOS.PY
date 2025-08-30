# Treinando_CONSELHOS.py
# Treinamento de um modelo spaCy para reconhecer entidades CRM, CREFITO, COREN, CRO, CRP, CRFa

import os
import spacy
from spacy.tokens import DocBin
import json
import subprocess
from pathlib import Path
import re
import random

# ================================================================
# Função para extrair texto de forma segura
# ================================================================
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

# ================================================================
# Função para ajustar anotações de CRM / CREFITO / outros
# ================================================================
def ajustar_anotacoes_conselho(texto):
    padroes = {
        "CRM": re.compile(r"\bCRM\s*[-:]?\s*\d{2,6}(?:-\w{1,3})?\b", re.IGNORECASE),
        "CREFITO": re.compile(r"\bCREFITO\s*[-:]?\s*\d{1,6}(?:/\w{1,3})?\b", re.IGNORECASE),
        "COREN": re.compile(r"\bCOREN\s*[-:]?\s*\d{2,6}(?:-\w{1,3})?\b", re.IGNORECASE),
        "CRO": re.compile(r"\bCRO\s*[-:]?\s*\d{2,6}(?:/\w{1,3})?\b", re.IGNORECASE),
        "CRP": re.compile(r"\bCRP\s*[-:]?\s*\d{2,6}(?:/\w{1,3})?\b", re.IGNORECASE),
        "CRFa": re.compile(r"\bCRFa\s*[-:]?\s*\d{2,6}\b", re.IGNORECASE)
    }

    novas_entidades = []
    for label, padrao in padroes.items():
        for match in padrao.finditer(texto):
            novas_entidades.append([match.start(), match.end(), label])
    return novas_entidades

# ================================================================
# Função para gerar dados sintéticos
# ================================================================
def gerar_dados_sinteticos():
    exemplos = {
        "CRM": ["CRM 12345-SP", "CRM-9876", "CRM: 112233"],
        "CREFITO": ["CREFITO-5 67890", "CREFITO 1234", "CREFITO: 998877"],
        "COREN": ["COREN 112233", "COREN-87654", "COREN: 554433"],
        "CRO": ["CRO 445566", "CRO-9988", "CRO: 223344"],
        "CRP": ["CRP-08/12345", "CRP 54321", "CRP: 112233"],
        "CRFa": ["CRFa 98765", "CRFa-1122", "CRFa: 334455"]
    }

    templates = [
        "Registro profissional: {}",
        "Atendido pelo especialista ({}).",
        "Número de inscrição: {}",
        "Documento de conselho: {}",
        "Profissional identificado: {}",
        "Conselho registrado: {}"
    ]

    dados_sinteticos = []
    for label, exemplos_label in exemplos.items():
        for exemplo in exemplos_label:
            for _ in range(5):
                template = random.choice(templates)
                texto = template.format(exemplo)
                start = texto.find(exemplo)
                end = start + len(exemplo)
                dados_sinteticos.append((texto, {"entities": [[start, end, label]]}))
    return dados_sinteticos

# ================================================================
# Função para carregar e ajustar os dados
# ================================================================
def carregar_dados(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    treinamentos = []
    for i, item in enumerate(data):
        if len(item) < 2:
            continue

        texto = extrair_texto(item)
        anotacao = item[1]

        entities = []
        if isinstance(anotacao, list):
            for elem in anotacao:
                if isinstance(elem, dict) and "entities" in elem:
                    entities = elem["entities"]
                    break
        elif isinstance(anotacao, dict) and "entities" in anotacao:
            entities = anotacao["entities"]

        entidades_ajustadas = []
        for ent in entities:
            if isinstance(ent, list) and len(ent) > 2 and ent[2] in ["CRM", "CREFITO", "COREN", "CRO", "CRP", "CRFa"]:
                entidades_ajustadas.append([ent[0], ent[1], ent[2]])

        regex_extra = ajustar_anotacoes_conselho(texto)
        entidades_ajustadas.extend(regex_extra)

        if entidades_ajustadas:
            treinamentos.append((texto, {"entities": entidades_ajustadas}))

    print(f"Total de itens processados em {filepath}: {len(treinamentos)}")
    if treinamentos:
        print(f"Exemplo: {treinamentos[0][0][:80]}...")
        print(f"Entidades: {treinamentos[0][1]['entities']}")
    return treinamentos

# ================================================================
# Carregar datasets
# ================================================================
base_train = carregar_dados("ner_treino_split.json")
base_dev = carregar_dados("ner_validacao_split.json")

# Dados sintéticos
dados_sinteticos = gerar_dados_sinteticos()
train_data = base_train + dados_sinteticos
dev_data = base_dev + dados_sinteticos

print(f"\nDados de treino: {len(train_data)}")
print(f"Dados de validação: {len(dev_data)}")

# ================================================================
# Inicializar pipeline spaCy
# ================================================================
nlp = spacy.blank("pt")
ner = nlp.add_pipe("ner", last=True)

# Adicionar todos os labels
for label in ["CRM", "CREFITO", "COREN", "CRO", "CRP", "CRFa"]:
    ner.add_label(label)

# ================================================================
# Criar DocBin (corrigido)
# ================================================================
def criar_docbin(data):
    doc_bin = DocBin()
    for texto, anotacao in data:
        if not isinstance(texto, str):
            continue
        doc = nlp.make_doc(texto)
        spans = []
        for ent in anotacao["entities"]:
            if len(ent) < 3:
                continue
            start, end, label = ent
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                spans.append(span)

        # 🔹 Remover spans duplicados e sobrepostos
        spans = spacy.util.filter_spans(spans)

        doc.ents = spans
        doc_bin.add(doc)
    return doc_bin

criar_docbin(train_data).to_disk("train_CONSELHOS.spacy")
criar_docbin(dev_data).to_disk("dev_CONSELHOS.spacy")

# ================================================================
# Treinamento com spaCy CLI
# ================================================================
caminho_modelo = os.path.join(os.getcwd(), "modelo_NER_CONSELHOS")

# Gera config automática
subprocess.run([
    "spacy", "init", "config",
    "--lang", "pt",
    "--pipeline", "ner",
    "--force",
    "config.cfg"
])

# Treina modelo
subprocess.run([
    "spacy", "train",
    "config.cfg",
    "--output", caminho_modelo,
    "--paths.train", "train_CONSELHOS.spacy",
    "--paths.dev", "dev_CONSELHOS.spacy"
])

print(f"\n✅ Modelo treinado salvo em: {caminho_modelo}")
