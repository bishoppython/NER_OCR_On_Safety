import os
import spacy
from spacy.tokens import DocBin
import json
import subprocess
import re
import random
from datetime import datetime, timedelta
import locale
from faker import Faker

fake = Faker('pt_BR')  # inicializar Faker com localidade brasileira

# Configurar locale para português
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except:
    locale.setlocale(locale.LC_TIME, 'pt_BR')

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


# Padrões regex para TIPO_DOC (corrigido)
padrao_documento = re.compile(
    r'\b(?:'
    r'RELAT[ÓO]RIO\s*M[ÉE]DICO|'
    r'DECLARA[CÇ][ÃA]O\s*M[ÉE]DICA?|'
    r'ATESTADO\s*M[ÉE]DICO|'
    r'RECEITU[AÁ]RIO\s*M[ÉE]DICO|'
    r'DECLARACAO|'  # sem acento
    r'ATESTADO'      # sem "médico"
    r')\b',
    re.IGNORECASE
)

# Função para gerar dados sintéticos (corrigida)
def gerar_dados_sinteticos_documento(num_exemplos=500):
    documentos = [
        "RELATORIO MEDICO",
        "DECLARAÇÃO", "ATESTADO",
    ]
    
    templates = [
        "{documento} Atesto que o paciente",
        "Clinica hospitalar \n {documento}",
        "{documento} atesto para os devidos fins",
        "{documento} declaro que o paciente",
        "Hospital das Clinicas \n {documento} declaro que sr(a).",
        "{documento} Confirmo que o paciente",
        "{documento} declaro para os devidos fins que o paciente"
    ]
    
    dados_sinteticos = []
    for _ in range(num_exemplos):
        # Exemplo positivo
        doc = random.choice(documentos)
        template = random.choice(templates)
        texto = template.format(documento=doc)
        
        entities = []
        doc_match = re.search(re.escape(doc), texto, re.IGNORECASE)
        if doc_match:
            start, end = doc_match.span()
            entities.append([start, end, "TIPO_DOC"])
        
        dados_sinteticos.append((texto, {'entities': entities}))
        
        # Exemplo negativo (sem entidades)
        texto_negativo = f"Paciente {fake.name()} compareceu para consulta"
        dados_sinteticos.append((texto_negativo, {'entities': []}))
    
    return dados_sinteticos

# Função para ajustar anotações de documento
def ajustar_anotacoes_documento(texto):
    entities = []
    for match in padrao_documento.finditer(texto):
        start, end = match.span()
        entities.append([start, end, "TIPO_DOC"])
    return entities


# Função para carregar e ajustar os dados para documento
def carregar_dados_documento(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    treinamentos = []
    
    for i, item in enumerate(data):
        if len(item) < 2:
            continue
            
        texto = extrair_texto(item)
        entities = ajustar_anotacoes_documento(texto)
        
        if entities:
            treinamentos.append((texto, {'entities': entities}))
    
    print(f"Total de itens processados para TIPO_DOC: {len(treinamentos)}")
    return treinamentos

# Função para criar DocBin
def criar_docbin(nlp, data, labels):
    doc_bin = DocBin()
    for texto, anotacao in data:
        if not isinstance(texto, str):
            continue
            
        doc = nlp.make_doc(texto)
        ents = []
        for ent in anotacao["entities"]:
            if len(ent) < 3:
                continue
            start, end, label = ent[0], ent[1], ent[2]
            if label in labels:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
        doc.ents = ents
        doc_bin.add(doc)
    return doc_bin

# Função para treinar modelo
def treinar_modelo(tipo, train_data, dev_data, label):
    # Diretório para salvar o modelo
    caminho_modelo = f"modelo_NER_{tipo}"
    
    # Configuração otimizada para CPU
    config_content = f"""
    [paths]
    train = "train_{tipo}.spacy"
    dev = "dev_{tipo}.spacy"
    
    [system]
    gpu_allocator = null
    
    [nlp]
    lang = "pt"
    pipeline = ["tok2vec","ner"]
    
    [components]
    
    [components.tok2vec]
    factory = "tok2vec"
    
    [components.tok2vec.model]
    @architectures = "spacy.Tok2Vec.v2"
    
    [components.tok2vec.model.embed]
    @architectures = "spacy.MultiHashEmbed.v2"
    width = 96
    attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
    rows = [5000,1000,2500,2500]
    include_static_vectors = false
    
    [components.tok2vec.model.encode]
    @architectures = "spacy.MaxoutWindowEncoder.v2"
    width = 96
    depth = 4
    window_size = 1
    maxout_pieces = 3
    
    [components.ner]
    factory = "ner"
    
    [components.ner.model]
    @architectures = "spacy.TransitionBasedParser.v2"
    state_type = "ner"
    extra_state_tokens = false
    hidden_width = 64
    maxout_pieces = 2
    use_upper = true
    nO = null
    
    [components.ner.model.tok2vec]
    @architectures = "spacy.Tok2VecListener.v1"
    width = ${{components.tok2vec.model.encode.width}}
    
    [corpora]
    
    [corpora.train]
    @readers = "spacy.Corpus.v1"
    path = ${{paths.train}}
    
    [corpora.dev]
    @readers = "spacy.Corpus.v1"
    path = ${{paths.dev}}
    
    [training]
    dev_corpus = "corpora.dev"
    train_corpus = "corpora.train"
    seed = 42
    gpu_allocator = null
    accumulate_gradient = 1
    patience = 1600
    max_epochs = 50
    max_steps = 20000
    
    [training.batcher]
    @batchers = "spacy.batch_by_words.v1"
    discard_oversize = false
    tolerance = 0.2
    size = 1000
    
    [training.optimizer]
    @optimizers = "Adam.v1"
    beta1 = 0.9
    beta2 = 0.999
    L2_is_weight_decay = true
    L2 = 0.01
    grad_clip = 1.0
    use_averages = false
    eps = 0.00000001
    learn_rate = 0.001
    
    [training.logger]
    @loggers = "spacy.ConsoleLogger.v1"
    progress_bar = false
    
    [initialize]
    vectors = null
    """
    
    # Salvar configuração
    config_path = f"config_{tipo}.cfg"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    # Inicializar pipeline
    nlp = spacy.blank("pt")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    ner.add_label(label)
    
    # Criar DocBins
    criar_docbin(nlp, train_data, [label]).to_disk(f"train_{tipo}.spacy")
    criar_docbin(nlp, dev_data, [label]).to_disk(f"dev_{tipo}.spacy")
    
    # Treinar o modelo
    subprocess.run([
        "spacy", "train",
        config_path,
        "--output", caminho_modelo,
        "--gpu-id", "-1"  # Forçar uso de CPU
    ])
    
    print(f"\nModelo treinado para {tipo} salvo em: {caminho_modelo}")
    return caminho_modelo

# =======================
# Treino TIPO_DOC
# =======================
print("\n" + "="*50)
print("INICIANDO TREINAMENTO PARA TIPO_DOC")
print("="*50)

# Carregar dados
base_train_data = carregar_dados_documento("ner_treino_split.json")
base_dev_data = carregar_dados_documento("ner_validacao_split.json")

# Gerar dados sintéticos
dados_sinteticos_doc = gerar_dados_sinteticos_documento(800)

# Combinar dados
train_data_doc = base_train_data + dados_sinteticos_doc
dev_data_doc = base_dev_data + dados_sinteticos_doc[:200]  # Usar apenas parte para validação

print(f"Dados de treino para TIPO_DOC: {len(train_data_doc)}")
print(f"Dados de validação para TIPO_DOC: {len(dev_data_doc)}")

# Treinar modelo TIPO_DOC
caminho_modelo_doc = treinar_modelo(
    tipo="DOCUMENTO",
    train_data=train_data_doc,
    dev_data=dev_data_doc,
    label="TIPO_DOC"
)

print("\n" + "="*50)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*50)
print(f"Modelo para tipos de documento: {caminho_modelo_doc}")