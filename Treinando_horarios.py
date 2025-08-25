import os
import spacy
from spacy.tokens import DocBin
import json
import subprocess
import re
import random
from datetime import time, timedelta

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

# Função para gerar horários aleatórios
def gerar_horario_aleatorio(manha=False):
    hora = random.randint(8 if manha else 13, 11 if manha else 22)
    minuto = random.choice([0, 15, 30, 45])
    return f"{hora:02d}:{minuto:02d}"

# Função para gerar dados sintéticos de horários
def gerar_dados_sinteticos_horario():
    # Formatos para dois horários
    formatos_dois = [
        "Atendimento das {} às {}",
        "Horário: {} - {}",
        "Funcionamento: {} até {}",
        "Aberto de {} a {}",
        "Das {} as {}",
        "{} até às {}",
        "Atendemos das {} às {}",
        "Expediente: {} a {}",
        "Segunda a sexta: {} - {}",
        "{} até {}"
    ]
    
    # Formatos para um horário
    formatos_um = [
        "Horário de funcionamento: {}",
        "Atendimento: {}",
        "Disponível: {}",
        "Plantão: {}",
        "Serviço: {}",
        "Abertura: {}",
        "Fechamento: {}",
        "Início: {}",
        "Término: {}",
        "Até {}",
        "A partir de {}"
    ]
    
    dados_sinteticos = []
    for _ in range(200):  # Gerar 200 exemplos sintéticos
        # Gerar exemplos com dois horários
        formato = random.choice(formatos_dois)
        inicio = gerar_horario_aleatorio(manha=True)
        fim = gerar_horario_aleatorio()
        
        texto = formato.format(inicio, fim)
        
        # Encontrar posições dos horários
        start_inicio = texto.find(inicio)
        end_inicio = start_inicio + len(inicio)
        
        start_fim = texto.find(fim)
        end_fim = start_fim + len(fim)
        
        if start_inicio == -1 or start_fim == -1:
            continue
            
        dados_sinteticos.append((
            texto, 
            {
                'entities': [
                    [start_inicio, end_inicio, "HORARIO_INICIO_ATENDIMENTO"],
                    [start_fim, end_fim, "HORARIO_FIM_ATENDIMENTO"]
                ]
            }
        ))
        
        # Gerar exemplos com apenas um horário (30% de chance)
        if random.random() > 0.7:
            formato_single = random.choice(formatos_um)
            texto_single = formato_single.format(inicio)
            start = texto_single.find(inicio)
            end = start + len(inicio)
            
            if start != -1:
                label = random.choice(["HORARIO_INICIO_ATENDIMENTO", "HORARIO_FIM_ATENDIMENTO"])
                dados_sinteticos.append((
                    texto_single,
                    {'entities': [[start, end, label]]}
                ))
    
    return dados_sinteticos

# Função para carregar e ajustar os dados (suporta JSON e JSONL)
def carregar_dados_horarios(filepath):
    data = []
    
    # Verificar se é arquivo JSONL (.jsonl)
    if filepath.endswith('.jsonl'):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Erro ao decodificar linha: {line}\nErro: {e}")
    else:  # Assume formato JSON padrão
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Erro ao carregar JSON: {e}")
            return []
    
    treinamentos = []
    
    for i, item in enumerate(data):
        if not isinstance(item, list) or len(item) < 2:
            print(f"Item {i} inválido: não é uma lista ou não possui elementos suficientes")
            print(f"Conteúdo do item: {item}")
            continue
            
        texto = extrair_texto(item)
        anotacao = item[1]
        
        entities = []
        if isinstance(anotacao, list):
            for elem in anotacao:
                if isinstance(elem, dict) and 'entities' in elem:
                    entities = elem['entities']
                    break
                elif isinstance(elem, list):
                    # Formato direto: lista de entidades
                    entities = elem
        elif isinstance(anotacao, dict) and 'entities' in anotacao:
            entities = anotacao['entities']
        elif isinstance(anotacao, list):
            # Formato direto: lista de entidades
            entities = anotacao
        
        entidades_validas = []
        for ent in entities:
            if isinstance(ent, list) and len(ent) >= 3:
                # Aceitar apenas entidades de horário
                if ent[2] in ['HORARIO_INICIO_ATENDIMENTO', 'HORARIO_FIM_ATENDIMENTO']:
                    entidades_validas.append(ent)
            elif isinstance(ent, dict) and 'label' in ent and 'start' in ent and 'end' in ent:
                # Formato alternativo: dicionário com chaves label/start/end
                if ent['label'] in ['HORARIO_INICIO_ATENDIMENTO', 'HORARIO_FIM_ATENDIMENTO']:
                    entidades_validas.append([ent['start'], ent['end'], ent['label']])
        
        if entidades_validas:
            treinamentos.append((texto, {'entities': entidades_validas}))
        else:
            print(f"Item {i} não possui entidades válidas: {entities}")
    
    print(f"Total de itens processados em {filepath}: {len(treinamentos)}")
    return treinamentos

# Carregar dados de treino e validação
base_train = carregar_dados_horarios("horarios_train_data/labeled_dataset_horarios_corrigido.json")
base_dev = carregar_dados_horarios("horarios_train_data/spacy_dataset_horarios_dev.jsonl")

# Gerar dados sintéticos
dados_sinteticos = gerar_dados_sinteticos_horario()

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

# Adicionar rótulos para horários
ner.add_label("HORARIO_INICIO_ATENDIMENTO")
ner.add_label("HORARIO_FIM_ATENDIMENTO")

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
criar_docbin(train_data).to_disk("train_horarios.spacy")
criar_docbin(dev_data).to_disk("dev_horarios.spacy")

# Obter o diretório atual do script
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Criar caminho para salvar o modelo
caminho_modelo = os.path.join(diretorio_atual, "modelo_NER_horarios")

# Gera o arquivo de configuração
subprocess.run([
    "spacy", "init", "config",
    "--lang", "pt",
    "--pipeline", "ner",
    "--force",
    "config_horarios.cfg"
])

# Treinar o modelo
subprocess.run([
    "spacy", "train",
    "config_horarios.cfg",
    "--output", caminho_modelo,
    "--paths.train", "train_horarios.spacy",
    "--paths.dev", "dev_horarios.spacy"
])

print(f"Modelo treinado salvo em: {caminho_modelo}")