# -*- coding: utf-8 -*-
"""
Treinamento de NER (spaCy) focado apenas em TEMPO_AFASTAMENTO.

Arquivos esperados no mesmo diretório:
- ner_treino_split.json
- ner_validacao_split.json

Saídas:
- train_TEMPO.spacy, dev_TEMPO.spacy
- diretório do modelo: modelo_NER_TEMPO_AFASTAMENTO/
"""

import os
import re
import json
import random
import subprocess
import spacy
from spacy.tokens import DocBin

# -------------------------------
# Utilidades
# -------------------------------
def extrair_texto(item):
    """Extrai o texto do item (string ou lista/objetos) com tolerância."""
    texto = item[0]
    if isinstance(texto, str):
        return texto
    if isinstance(texto, list):
        partes = []
        for el in texto:
            if isinstance(el, str):
                partes.append(el)
            elif isinstance(el, dict) and 'text' in el:
                partes.append(el['text'])
            else:
                partes.append(str(el))
        return " ".join(partes)
    return str(texto)

# Padrões para durações: números + unidade, com variações e acentos
UNIDADES = r"(?:dia(?:s)?|hora(?:s)?|semana(?:s)?|m[êe]s(?:es)?)"
NUMERO_VARIANTE = r"\d{1,3}(?:\s*\([\w\.]+?\))?"           # ex: 1 (Um.)
PADRAO_NUM_UNIDADE = rf"\b{NUMERO_VARIANTE}\s*{UNIDADES}\b"

# Alguns termos textuais comuns em atestados
PADROES_TEXTUAIS = [
    r"per[ií]odo da manh[ãa]",
    r"per[ií]odo da tarde",
    r"per[ií]odo da noite",
    r"turno da manh[ãa]",
    r"turno da tarde",
    r"turno da noite",
    r"per[ií]odo integral"
]

PADRAO_TEXTO = re.compile("|".join(PADROES_TEXTUAIS), flags=re.IGNORECASE)

# Âncora: “afastamento de ...”
PADRAO_ANCORA = re.compile(
    r"afastamento\s+de\s+(.{1,80}?)"               # captura até ~80 chars
    r"(?:\s+de\s+suas\s+atividades|[\.\n]|$)",     # fecha no marcador típico
    flags=re.IGNORECASE | re.DOTALL
)

PADRAO_NUM_UNID_RE = re.compile(PADRAO_NUM_UNIDADE, flags=re.IGNORECASE)

def localizar_span(texto, trecho):
    """Retorna (start, end) do primeiro match exato de 'trecho' em 'texto'."""
    start = texto.find(trecho)
    if start == -1:
        return None
    return (start, start + len(trecho))

def corrigir_ou_inferir_tempo(texto):
    """
    Tenta inferir o TEMPO_AFASTAMENTO a partir da âncora 'afastamento de ...',
    priorizando padrões numéricos + unidade; se não houver, aceita textual.
    Retorna lista [[start, end, 'TEMPO_AFASTAMENTO']] (0 ou 1 item).
    """
    m = PADRAO_ANCORA.search(texto)
    if not m:
        return []

    janela = m.group(1)
    # 1) Prioriza número + unidade (ex: '10 dias', '1 (Um.) dias', '5 horas')
    m_num = PADRAO_NUM_UNID_RE.search(janela)
    if m_num:
        trecho = m_num.group(0)
        pos = localizar_span(texto, trecho)
        if pos:
            return [[pos[0], pos[1], "TEMPO_AFASTAMENTO"]]

    # 2) Se não há número, aceita textual (ex: 'período da tarde')
    m_tx = PADRAO_TEXTO.search(janela)
    if m_tx:
        trecho = m_tx.group(0)
        pos = localizar_span(texto, trecho)
        if pos:
            return [[pos[0], pos[1], "TEMPO_AFASTAMENTO"]]

    # 3) Último recurso: use a janela inteira (limitando ruído)
    trecho_fallback = janela.strip()
    # evita trechos gigantescos
    if 1 <= len(trecho_fallback) <= 60:
        pos = localizar_span(texto, trecho_fallback)
        if pos:
            return [[pos[0], pos[1], "TEMPO_AFASTAMENTO"]]
    return []

def carregar_dados(filepath):
    """
    Lê o JSON, mantém/ajusta APENAS a entidade TEMPO_AFASTAMENTO.
    Se o label não existir no item, tenta inferir pela âncora regex.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    saida = []
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

        ents_tempo = []
        # 1) coleta as spans existentes de TEMPO_AFASTAMENTO
        for ent in entities:
            if isinstance(ent, list) and len(ent) >= 3 and ent[2] == "TEMPO_AFASTAMENTO":
                # pequeno sanity-check: a substring deve parecer duração
                trecho = texto[ent[0]:ent[1]]
                if PADRAO_NUM_UNID_RE.search(trecho) or PADRAO_TEXTO.search(trecho):
                    ents_tempo.append(ent)
                else:
                    # repara via âncora se o rótulo original é suspeito
                    reparo = corrigir_ou_inferir_tempo(texto)
                    if reparo:
                        ents_tempo.extend(reparo)
            # ignora outras entidades

        # 2) se não havia label, tenta inferir
        if not ents_tempo:
            inferidas = corrigir_ou_inferir_tempo(texto)
            ents_tempo.extend(inferidas)

        if ents_tempo:
            saida.append((texto, {"entities": ents_tempo}))

    return saida

# -------------------------------
# Dados sintéticos (robustez)
# -------------------------------
def gerar_dados_sinteticos(n=80):
    quantias = ["1", "2", "3", "5", "7", "8", "9", "10", "11", "14", "15", "21", "30", "60"]
    unidades = ["dia", "dias", "hora", "horas", "semana", "semanas", "mês", "meses"]
    forma_var = [
        "{q} {u}",
        "{q} ({ql}) {u}",
        "{q}\n {u}",  # quebra de linha para simular ruído
    ]
    textos = [
        "Recomendado afastamento de {dur} de suas atividades.",
        "Afastamento de {dur}.",
        "Paciente deverá cumprir afastamento de {dur}.",
        "Sugere-se afastamento de {dur} para recuperação."
    ]
    por_extenso = {
        "1": "Um", "2": "Dois", "3": "Três", "5": "Cinco", "7": "Sete",
        "8": "Oito", "9": "Nove", "10": "Dez", "11": "Onze", "14": "Catorze",
        "15": "Quinze", "21": "Vinte e um", "30": "Trinta", "60": "Sessenta"
    }

    dados = []
    for _ in range(n):
        q = random.choice(quantias)
        u = random.choice(unidades)
        var = random.choice(forma_var)
        dur = var.format(q=q, ql=por_extenso[q], u=u)
        temp = random.choice(textos).format(dur=dur)
        start = temp.find(dur)
        if start == -1:
            continue
        end = start + len(dur)
        dados.append((temp, {"entities": [[start, end, "TEMPO_AFASTAMENTO"]]}))
    # alguns textuais
    textuais = [
        "Recomendado afastamento de período da tarde.",
        "Afastamento de período integral.",
        "Paciente deverá cumprir afastamento de turno da noite."
    ]
    for t in textuais:
        m = PADRAO_TEXTO.search(t)
        if m:
            dados.append((t, {"entities": [[m.start(), m.end(), "TEMPO_AFASTAMENTO"]]}))
    return dados

# -------------------------------
# Pipeline de treino
# -------------------------------
if __name__ == "__main__":
    # Carrega bases
    base_train = carregar_dados("ner_treino_split.json")
    base_dev = carregar_dados("ner_validacao_split.json")

    # Aumento sintético
    sint = gerar_dados_sinteticos(n=120)
    train_data = base_train + sint
    dev_data = base_dev + sint[: max(60, len(sint)//3)]

    print(f"Treino: {len(base_train)} originais + {len(sint)} sintéticos = {len(train_data)}")
    print(f"Validação: {len(base_dev)} originais + {len(dev_data)-len(base_dev)} sintéticos = {len(dev_data)}")

    # NER em branco (pt)
    nlp = spacy.blank("pt")
    ner = nlp.add_pipe("ner", last=True)
    ner.add_label("TEMPO_AFASTAMENTO")

    # DocBins
    def criar_docbin(data):
        db = DocBin()
        for texto, anot in data:
            doc = nlp.make_doc(texto)
            spans = []
            for start, end, label in anot["entities"]:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    spans.append(span)
            doc.ents = spans
            db.add(doc)
        return db

    criar_docbin(train_data).to_disk("train_TEMPO.spacy")
    criar_docbin(dev_data).to_disk("dev_TEMPO.spacy")

    # Config + treino
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_NER_TEMPO_AFASTAMENTO")

    subprocess.run([
        "spacy", "init", "config",
        "--lang", "pt", "--pipeline", "ner", "--force", "config_tempo.cfg"
    ], check=True)

    subprocess.run([
        "spacy", "train", "config_tempo.cfg",
        "--output", out_dir,
        "--paths.train", "train_TEMPO.spacy",
        "--paths.dev", "dev_TEMPO.spacy"
    ], check=True)

    print(f"✅ Modelo treinado em: {out_dir}")
