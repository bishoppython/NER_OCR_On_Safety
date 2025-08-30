# import spacy
# import random
# import json
# from spacy.tokens import DocBin
# from spacy.training.example import Example
# from pathlib import Path
# import copy

# # =============================================================================
# # Carregar datasets expandidos
# # =============================================================================
# def carregar_dados(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     treinamentos = []
#     for i, item in enumerate(data):
#         if len(item) < 2:
#             print(f"Item {i} inválido: não possui elementos suficientes")
#             continue

#         texto = item[0]
#         anotacao = item[1]

#         entities = []
#         if isinstance(anotacao, dict) and 'entities' in anotacao:
#             entities = anotacao['entities']

#         treinamentos.append((texto, {"entities": entities}))

#     print(f"Total de itens processados de {filepath}: {len(treinamentos)}")
#     return treinamentos

# # Usar os datasets expandidos
# treino_path = "nome_treino/ner_treino_nome_paciente_expandido.json"
# valid_path = "nome_treino/ner_validacao_nome_paciente_expandido.json"

# base_train = carregar_dados(treino_path)
# base_dev = carregar_dados(valid_path)

# train_data = base_train
# dev_data = base_dev

# print(f"\nDados de treino: {len(train_data)}")
# print(f"Dados de validação: {len(dev_data)}")

# # =============================================================================
# # Converter para DocBin (formato spaCy)
# # =============================================================================
# def criar_docbin(data, output_file):
#     nlp = spacy.blank("pt")
#     doc_bin = DocBin()

#     for item in data:
#         if isinstance(item, (list, tuple)):
#             texto, anotacao = item
#         else:
#             continue  # ignora formatos inválidos

#         if not isinstance(texto, str):
#             continue  # garante que só entra string

#         doc = nlp.make_doc(texto)
#         ents = []
#         for ent in anotacao.get("entities", []):
#             span = doc.char_span(ent[0], ent[1], label=ent[2])
#             if span is not None:
#                 ents.append(span)
#         doc.ents = ents
#         doc_bin.add(doc)

#     doc_bin.to_disk(output_file)
#     print(f"Arquivo {output_file} gerado com {len(doc_bin)} exemplos")


# # Gerar arquivos .spacy
# criar_docbin(train_data, "train_nome_paciente.spacy")
# criar_docbin(dev_data, "dev_nome_paciente.spacy")

# # =============================================================================
# # Treinamento do modelo
# # =============================================================================
# train_docbin = DocBin().from_disk("train_nome_paciente.spacy")
# dev_docbin = DocBin().from_disk("dev_nome_paciente.spacy")

# nlp = spacy.blank("pt")
# train_docs = list(train_docbin.get_docs(nlp.vocab))
# dev_docs = list(dev_docbin.get_docs(nlp.vocab))

# def prepare_examples(docs):
#     examples = []
#     for doc in docs:
#         entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
#         examples.append((doc.text, {"entities": entities}))
#     return examples

# train_examples = prepare_examples(train_docs)
# dev_examples = prepare_examples(dev_docs)

# # Configurar NER
# if "ner" not in nlp.pipe_names:
#     ner = nlp.add_pipe("ner", last=True)
# else:
#     ner = nlp.get_pipe("ner")

# for _, annotations in train_examples:
#     for ent in annotations["entities"]:
#         ner.add_label(ent[2])

# # =============================================================================
# # Funções de avaliação (Precision, Recall, F1)
# # =============================================================================
# def avaliar(nlp, validation_data):
#     tp, fp, fn = 0, 0, 0

#     for text, annotations in validation_data:
#         doc = nlp(text)
#         predicted = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
#         true = [(ent[0], ent[1], ent[2]) for ent in annotations.get("entities", [])]

#         for ent in predicted:
#             if ent in true:
#                 tp += 1
#             else:
#                 fp += 1
#         for ent in true:
#             if ent not in predicted:
#                 fn += 1

#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#     return precision, recall, f1, tp, fp, fn

# # =============================================================================
# # Loop de treinamento
# # =============================================================================
# optimizer = nlp.begin_training()
# best_f1 = 0.0
# patience = 10
# no_improvement = 0
# best_model = None

# print("\nIniciando treinamento...")
# print("=" * 60)

# for epoch in range(50):
#     random.shuffle(train_examples)
#     losses = {}
#     for text, annotations in train_examples:
#         doc = nlp.make_doc(text)
#         example = Example.from_dict(doc, annotations)
#         nlp.update([example], sgd=optimizer, losses=losses, drop=0.3)

#     precision, recall, f1, tp, fp, fn = avaliar(nlp, dev_examples)

#     print(f"Época {epoch+1}")
#     print(f"  Perda treino: {losses.get('ner', 0.0):.4f}")
#     print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
#     print(f"  TP={tp} | FP={fp} | FN={fn}")

#     if f1 > best_f1:
#         best_f1 = f1
#         best_model = copy.deepcopy(nlp)
#         no_improvement = 0
#         print("  ⭐ Novo melhor modelo!")
#     else:
#         no_improvement += 1
#         if no_improvement >= patience:
#             print(f"  ⛔ Parando cedo após {patience} épocas sem melhoria")
#             break

#     print("-" * 60)

# # =============================================================================
# # Salvar melhor modelo
# # =============================================================================
# output_dir = Path("modelo_NER_NOME_PACIENTE")
# if best_model:
#     nlp = best_model

# if not output_dir.exists():
#     output_dir.mkdir()

# nlp.to_disk(output_dir)
# print("\n" + "=" * 60)
# print(f"✅ Treinamento finalizado!")
# print(f"Melhor modelo salvo em: {output_dir}")
# print(f"Melhor F1 na validação: {best_f1:.4f}")
# print("=" * 60)


import spacy
import random
import json
from spacy.tokens import DocBin
from spacy.training.example import Example
from pathlib import Path
import copy
import re

# =============================================================================
# Função para carregar dataset original
# =============================================================================
def carregar_dados(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    treinamentos = []
    for i, item in enumerate(data):
        if not isinstance(item, list) or len(item) < 2:
            print(f"Item {i} inválido: {item}")
            continue

        texto, anotacao = item[0], item[1]
        if not isinstance(texto, str) or not isinstance(anotacao, dict):
            print(f"Item {i} inválido (texto ou anotação incorretos): {item}")
            continue

        entities = anotacao.get("entities", [])
        treinamentos.append((texto, {"entities": entities}))

    print(f"Total de itens processados de {filepath}: {len(treinamentos)}")
    return treinamentos

# =============================================================================
# Função para criar dados sintéticos diversificados
# =============================================================================
def gerar_dados_sinteticos(base_data, n_variacoes=3):
    titulos = ["ATESTADO", "DECLARAÇÃO", "RELATÓRIO MÉDICO", "LAUDO MÉDICO", "ATESTO"]
    prefixos = ["o(a) paciente", "paciente", "Paciente:", "Identifico o paciente", "Nome do paciente"]
    pronomes = ["Sr.", "Sra.", "Dr.", "Dra.", ""]
    clinicas = ["Clínica Santa Maria", "Hospital São João", "Laboratório Vida", "Clínica Saúde"]

    novos = []
    for texto, anotacao in base_data:
        for ent in anotacao["entities"]:
            start, end, label = ent
            nome = texto[start:end]

            for _ in range(n_variacoes):
                titulo = random.choice(titulos)
                prefixo = random.choice(prefixos)
                pronome = random.choice(pronomes)
                clinica = random.choice(clinicas)
                dia = f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/2024"
                hora = f"{random.randint(8,17):02d}:{random.choice([0,30]):02d}"

                novo_texto = f"{titulo}\n{prefixo} {pronome} {nome} esteve em atendimento na {clinica} em {dia} às {hora}."
                novo_start = novo_texto.find(nome)
                novo_end = novo_start + len(nome)

                novos.append((novo_texto, {"entities": [(novo_start, novo_end, label)]}))
    return novos

# =============================================================================
# Função para validar spans antes de adicionar ao DocBin
# =============================================================================
def is_valid_span(text, start, end):
    span = text[start:end]
    if re.match(r"\d{2}/\d{2}/\d{4}", span): return False
    if re.match(r"\d{2}:\d{2}", span): return False
    if any(c in span.lower() for c in ["clínica", "hospital", "laboratório"]): return False
    return True

# =============================================================================
# Função para criar DocBin
# =============================================================================
def criar_docbin(data, output_file):
    nlp = spacy.blank("pt")
    doc_bin = DocBin()
    for item in data:
        # Suporta listas e tuplas
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
            texto, anotacao = item
        else:
            print(f"⚠️ Ignorando exemplo inválido (não é string): {item}")
            continue

        doc = nlp.make_doc(texto)
        spans = [doc.char_span(start, end, label) for start, end, label in anotacao.get("entities", [])]
        spans = [s for s in spans if s is not None and is_valid_span(texto, s.start_char, s.end_char)]
        doc.ents = spans
        doc_bin.add(doc)

    doc_bin.to_disk(output_file)
    print(f"Arquivo {output_file} gerado com {len(doc_bin)} exemplos")

# =============================================================================
# Carregar datasets
# =============================================================================
treino_path = "nome_treino/ner_treino_nome_paciente_expandido.json"
valid_path = "nome_treino/ner_validacao_nome_paciente_expandido.json"

base_train = carregar_dados(treino_path)
base_dev = carregar_dados(valid_path)

# Gerar dados sintéticos e adicionar ao treino
dados_sinteticos = gerar_dados_sinteticos(base_train, n_variacoes=5)
train_data = base_train + dados_sinteticos
dev_data = base_dev

print(f"Treino original: {len(base_train)} | Dados sintéticos: {len(dados_sinteticos)}")
print(f"Total treino após expansão: {len(train_data)} | Validação: {len(dev_data)}")

# Criar arquivos DocBin
criar_docbin(train_data, "train_nome_paciente.spacy")
criar_docbin(dev_data, "dev_nome_paciente.spacy")

# =============================================================================
# Treinamento
# =============================================================================
nlp = spacy.blank("pt")

# Criar NER primeiro
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Adicionar EntityRuler antes do NER
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.get_pipe("entity_ruler")

patterns = [
    {"label": "DATA", "pattern": [{"TEXT": {"REGEX": r"\d{2}/\d{2}/\d{4}"}}]},
    {"label": "HORA", "pattern": [{"TEXT": {"REGEX": r"\d{2}:\d{2}"}}]},
    {"label": "CLINICA", "pattern": [{"LOWER": {"IN": ["clinica", "hospital", "laboratório"]}}]}
]
ruler.add_patterns(patterns)

# Carregar DocBin
train_docbin = DocBin().from_disk("train_nome_paciente.spacy")
dev_docbin = DocBin().from_disk("dev_nome_paciente.spacy")
train_docs = list(train_docbin.get_docs(nlp.vocab))
dev_docs = list(dev_docbin.get_docs(nlp.vocab))

# Preparar exemplos para treinamento
def prepare_examples(docs):
    examples = []
    for doc in docs:
        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        examples.append((doc.text, {"entities": ents}))
    return examples

train_examples = prepare_examples(train_docs)
dev_examples = prepare_examples(dev_docs)

# Adicionar labels do NER
for _, ann in train_examples:
    for ent in ann["entities"]:
        ner.add_label(ent[2])

optimizer = nlp.begin_training()
best_f1 = 0.0
patience, no_improvement = 10, 0
best_model = None

print("\nIniciando treinamento...")
print("=" * 60)

for epoch in range(50):
    random.shuffle(train_examples)
    losses = {}
    for text, annotations in train_examples:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer, losses=losses, drop=0.3)

    # Avaliação simples
    tp, fp, fn = 0, 0, 0
    for text, ann in dev_examples:
        doc = nlp(text)
        pred = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        true = ann["entities"]
        tp += sum([1 for e in pred if e in true])
        fp += sum([1 for e in pred if e not in true])
        fn += sum([1 for e in true if e not in pred])
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Época {epoch+1} | Loss: {losses.get('ner',0):.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = copy.deepcopy(nlp)
        no_improvement = 0
        print("  ⭐ Novo melhor modelo salvo!")
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print("⛔ Parada antecipada por falta de melhoria")
            break

# Salvar modelo
output_dir = Path("modelo_NER_NOME_PACIENTE")
if best_model:
    best_model.to_disk(output_dir)
    print(f"\n✅ Modelo final salvo em {output_dir} com F1={best_f1:.4f}")

