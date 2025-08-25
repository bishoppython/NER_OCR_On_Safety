import os
import spacy
from spacy.tokens import DocBin
import json
import subprocess
import re
import random
from datetime import datetime, timedelta
import locale

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

# Padrões regex para datas
padrao_data = re.compile(
    r'(\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b)|'  # DD/MM/AAAA, DD.MM.AA, etc
    r'(\b\d{1,2}\s+de\s+[a-zç]{3,9}\s+de\s+\d{4}\b)|'  # 15 de setembro de 2025
    r'(\b(?:Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)[a-z]*[-/]\d{4}\b)|'  # Set-2025, Out/2025
    r'(\b\d{1,2}[-/](?:Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)[a-z]*[-/]\d{4}\b)',  # 15/Set/2025
    re.IGNORECASE
)

# Função para ajustar anotações de data
def ajustar_anotacoes_data(texto):
    entities = []
    for match in padrao_data.finditer(texto):
        start, end = match.span()
        entities.append([start, end, "DATA"])
    return entities

# Função para gerar dados sintéticos de data
def gerar_dados_sinteticos_data(num_exemplos=500):
    dados_sinteticos = []
    
    formatos_data = [
        "%d/%m/%Y",     # 07/07/2025
        "%d-%m-%Y",     # 07-07-2025
        "%d.%m.%Y",     # 07.07.2025
        "%d/%m/%y",     # 07/07/25
        "%Y/%m/%d",     # 2025/07/07
        "%d de %B de %Y", # 7 de julho de 2025
        "%d-%b-%Y",     # 07-Jul-2025
        "%b-%Y",        # Jul-2025
        "%B %Y",        # Julho 2025
        "%d/%b/%Y",     # 07/Jul/2025
    ]
    
    templates = [
        "Consulta agendada para {data}",
        "Atendimento realizado em {data}",
        "Em {data}, o paciente foi examinado",
        "Data do procedimento: {data}",
        "Registrado em {data}",
        "Comparecimento em {data}",
        "Dia {data}, foram realizados os exames",
        "Marcado para {data}",
        "Ocorrido em {data}",
        "Data de nascimento: {data}",
    ]
    
    for _ in range(num_exemplos):
        data_base = datetime.now() + timedelta(days=random.randint(-365, 365))
        formato_data = random.choice(formatos_data)
        data_str = data_base.strftime(formato_data)
        
        template = random.choice(templates)
        texto = template.format(data=data_str)
        
        entities = []
        data_match = re.search(re.escape(data_str), texto)
        if data_match:
            entities.append([data_match.start(), data_match.end(), "DATA"])
        
        dados_sinteticos.append((texto, {'entities': entities}))
    
    return dados_sinteticos

# Função para carregar e ajustar os dados para data
def carregar_dados_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    treinamentos = []
    
    for i, item in enumerate(data):
        if len(item) < 2:
            continue
            
        texto = extrair_texto(item)
        entities = ajustar_anotacoes_data(texto)
        
        if entities:
            treinamentos.append((texto, {'entities': entities}))
    
    print(f"Total de itens processados para DATA: {len(treinamentos)}")
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

# ============
# Treino DATA
# ============
print("\n" + "="*50)
print("INICIANDO TREINAMENTO PARA DATAS")
print("="*50)

# Carregar dados
base_train_data = carregar_dados_data("ner_treino_split.json")
base_dev_data = carregar_dados_data("ner_validacao_split.json")

# Gerar dados sintéticos
dados_sinteticos_data = gerar_dados_sinteticos_data(800)

# Combinar dados
train_data_data = base_train_data + dados_sinteticos_data
dev_data_data = base_dev_data + dados_sinteticos_data

print(f"Dados de treino para DATA: {len(train_data_data)}")
print(f"Dados de validação para DATA: {len(dev_data_data)}")

# Treinar modelo DATA
caminho_modelo_data = treinar_modelo(
    tipo="DATA",
    train_data=train_data_data,
    dev_data=dev_data_data,
    label="DATA"
)

print("\n" + "="*50)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*50)
print(f"Modelo para datas: {caminho_modelo_data}")