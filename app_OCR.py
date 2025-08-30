import spacy
import json
import re

# Função para validar e limpar datas
def limpar_data(texto_data):
    # Expressão regular para formatos de data válidos
    padrao_data = re.compile(
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|'          # DD/MM/AAAA ou DD/MM/AA
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b|'          # DD-MM-AAAA
        r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b|'        # DD.MM.AAAA
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|'          # DD/MM/AAAA
        r'\b\d{1,2}\s+de\s+[a-zç]{3,9}\s+de\s+\d{4}\b|'  # 15 de setembro de 2025
        r'\b\d{1,2}[-/](?:Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)[a-z]*[-/]\d{4}\b',  # 15/Set/2025
        re.IGNORECASE
    )
    
    # Buscar todas as datas válidas no texto
    datas_validas = padrao_data.findall(texto_data)
    if datas_validas:
        return datas_validas[0]  # Retorna a primeira data válida encontrada
    
    # Tentar extrair data de strings mais longas
    match = re.search(r'(\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b)', texto_data)
    if match:
        return match.group(1)
    
    return None

# Função para limpar e validar códigos CID
def processar_cid(texto_cid):
    cid_limpo = re.sub(r'^(CID[:\-]?\s*|\(|\))', '', texto_cid, flags=re.IGNORECASE)
    cid_limpo = cid_limpo.strip()
    if re.match(r'^[A-Z]\d+(\.\d+)?$', cid_limpo):
        return cid_limpo
    return None

# Função para validar o tipo de documento
def validar_tipo_documento(texto):
    padrao = re.compile(
        r'^(?:'
        r'RELAT[ÓO]RIO\s*M[ÉE]DICO|'    
        r'DECLARA[CÇ][ÃA]O\s*M[ÉE]DICA?|'
        r'ATESTADO\s*M[ÉE]DICO|'
        r'RECEITU[AÁ]RIO\s*M[ÉE]DICO|'
        r'LAUDO\s*M[ÉE]DICO|'
        r'DECLARACAO|'
        r'ATESTADO|'
        r'RELAT[ÓO]RIO|'
        r'DECLARA[CÇ][ÃA]O'
        r')$',
        re.IGNORECASE
    )
    return bool(padrao.match(texto.strip()))


def validar_nome_paciente(nome):
    # Verificar se contém prefixos médicos ou títulos
    if re.search(r'\b(?:Dr|Dra|Drª|Dr\.|Dra\.|CRM|CRF|Enf|Fisioter|Nutr)\b', nome, re.IGNORECASE):
        return False
    
    # Verificar se contém termos médicos ou administrativos
    termos_invalidos = [
        "afastamento", "indicado", "necessário", "tratamento", "diagnóstico", "repouso",
        "paciente", "compareceu", "atendimento", "consult", "avaliação", "clínica", "hospital",
        "unidade", "serviço", "período", "dias", "dia", "CID", "crm", "código", "documento",
        "declaro", "consta", "confirmo", "atesto", "realizou", "avaliado", "diagnosticado",
        "recomendado", "indicado", "necessario", "realizado", "acompanhamento", "cuidados",
        "dieta", "protocolo", "fins", "devidos", "fim", "inicio", "manhã", "tarde", "noite",
        "horário", "cpf", "laudo", "atestado", "declaração", "relatório", "receituário"
    ]
    
    if any(termo in nome.lower() for termo in termos_invalidos):
        return False
    
    # Verificar padrões inválidos
    padroes_invalidos = [
        r'\d',  # Números
        r'[.:;?!@#$%^&*()_+=|<>/\\{}\[\]~-]',  # Caracteres especiais
        r'\b(?:de|do|da|dos|das|e)\b',  # Preposições comuns
        r'\b(?:sr|sra|srta|sr\.|sra\.|srta\.)\b'  # Títulos de tratamento
    ]
    
    for padrao in padroes_invalidos:
        if re.search(padrao, nome, re.IGNORECASE):
            return False
    
    # Verificar estrutura do nome
    partes = nome.split()
    if len(partes) < 2:  # Deve ter pelo menos nome e sobrenome
        return False
    
    # Verificar partes inválidas
    partes_invalidas = ["cidade", "estado", "país", "rua", "avenida", "bairro", "nº", "número"]
    if any(parte in nome.lower() for parte in partes_invalidas):
        return False
    
    # Verificar se cada parte tem pelo menos 2 caracteres
    if any(len(parte) < 2 for parte in partes):
        return False
    
    # Verificar padrão de nome completo (primeira letra maiúscula em cada parte)
    if not all(parte[0].isupper() for parte in partes if len(parte) > 1):
        return False
    
    return True



# Carregar todos os modelos treinados individualmente por entidade
modelos = {
    "CID": spacy.load("modelo_treinado_CID/model-last"),
    "NOME_PACIENTE": spacy.load("modelo_NER_NOME_PACIENTE"),
    "DATA": spacy.load("modelo_NER_DATA/model-last"),   
    "TIPO_DOC": spacy.load("modelo_NER_DOCUMENTO/model-last"),
    "TEMPO_AFASTAMENTO": spacy.load("modelo_NER_TEMPO_AFASTAMENTO/model-last"),
    "CRM": spacy.load("modelo_NER_CONSELHOS/model-last"),
    "HORARIOS": spacy.load("modelo_NER_horarios/model-last"),
    # "HORARIO_INICIO_ATENDIMENTO": spacy.load("modelo_NER_HORARIOS/model-last"),
    # "HORARIO_FIM_ATENDIMENTO": spacy.load("modelo_NER_HORARIOS/model-last"),
}


# Entidades esperadas
ENTIDADES_ESPERADAS = [
    "NOME_PACIENTE", "CID", "DATA", "TIPO_DOC",
    "TEMPO_AFASTAMENTO", "CRM", 
    "HORARIO_INICIO_ATENDIMENTO", "HORARIO_FIM_ATENDIMENTO"
]

# def extrair_entidades_multimodelo(texto, modelos):
#     entidades = {ent: [] for ent in ENTIDADES_ESPERADAS}

#     for entidade, nlp_model in modelos.items():
#         doc = nlp_model(texto)
#         for ent in doc.ents:
#             texto_ent = ent.text.strip()
#             label_ent = ent.label_
            
#             # Processamento especial para CID
#             if entidade == "CID":
#                 cid_processado = processar_cid(texto_ent)
#                 if cid_processado and cid_processado not in entidades["CID"]:
#                     entidades["CID"].append(cid_processado)
            
#             # Processamento especial para DATA
#             elif entidade == "DATA":
#                 data_limpa = limpar_data(texto_ent)
#                 if data_limpa and data_limpa not in entidades["DATA"]:
#                     entidades["DATA"].append(data_limpa)
                    
#             # Processamento especial para TIPO_DOC
#             elif entidade == "TIPO_DOC":
#                 if validar_tipo_documento(texto_ent):
#                     entidades["TIPO_DOC"].append(texto_ent)
                    
#             # Processamento especial para NOME_PACIENTE
#             elif entidade == "NOME_PACIENTE":
#                 if validar_nome_paciente(texto_ent) and texto_ent not in entidades["NOME_PACIENTE"]:
#                     entidades["NOME_PACIENTE"].append(texto_ent)
            
#             # Processamento para outras entidades
#             else:
#                 if label_ent in entidades and texto_ent not in entidades[label_ent]:
#                     entidades[label_ent].append(texto_ent)

#     # =============================================
#     # PÓS-PROCESSAMENTO PARA NOMES DE PACIENTES
#     # =============================================
#     # 1. Remover nomes que são partes de outros nomes
#     nomes_finais = []
#     for nome in sorted(entidades["NOME_PACIENTE"], key=len, reverse=True):
#         # Verificar se este nome não é parte de um nome mais completo já na lista
#         if not any(nome != outro and nome in outro for outro in nomes_finais):
#             nomes_finais.append(nome)
    
#     # 2. Remover nomes que não atendem aos critérios mínimos
#     nomes_validos = [nome for nome in nomes_finais if validar_nome_paciente(nome)]
    
#     entidades["NOME_PACIENTE"] = nomes_validos

    #return entidades

def extrair_entidades_multimodelo(texto, modelos):
    entidades = {ent: [] for ent in ENTIDADES_ESPERADAS}
    
    for entidade, nlp_model in modelos.items():
        doc = nlp_model(texto)
        for ent in doc.ents:
            texto_ent = ent.text.strip()
            label_ent = ent.label_
            
            # Processamento especial para CID
            if entidade == "CID":
                cid_processado = processar_cid(texto_ent)
                if cid_processado and cid_processado not in entidades["CID"]:
                    entidades["CID"].append(cid_processado)
            
            # Processamento especial para DATA
            elif entidade == "DATA":
                data_limpa = limpar_data(texto_ent)
                if data_limpa and data_limpa not in entidades["DATA"]:
                    entidades["DATA"].append(data_limpa)
                    
            # Processamento especial para TIPO_DOC
            elif entidade == "TIPO_DOC":
                if validar_tipo_documento(texto_ent):
                    entidades["TIPO_DOC"].append(texto_ent)
                    
            # Processamento especial para NOME_PACIENTE
            # elif entidade == "NOME_PACIENTE":
            #     if validar_nome_paciente(texto_ent) and texto_ent not in entidades["NOME_PACIENTE"]:
            #         entidades["NOME_PACIENTE"].append(texto_ent)
            
            elif entidade == "NOME_PACIENTE":
                print(f"[DEBUG] Modelo NOME_PACIENTE encontrou: '{texto_ent}'")
                if texto_ent not in entidades["NOME_PACIENTE"]:
                    entidades["NOME_PACIENTE"].append(texto_ent)

            # Processamento para HORARIOS (agora unificado)
            elif entidade == "HORARIOS":
                if label_ent in ("HORARIO_INICIO_ATENDIMENTO", "HORARIO_INICIO"):
                    if texto_ent not in entidades["HORARIO_INICIO_ATENDIMENTO"]:
                        entidades["HORARIO_INICIO_ATENDIMENTO"].append(texto_ent)
                elif label_ent in ("HORARIO_FIM_ATENDIMENTO", "HORARIO_FIM"):
                    if texto_ent not in entidades["HORARIO_FIM_ATENDIMENTO"]:
                        entidades["HORARIO_FIM_ATENDIMENTO"].append(texto_ent)
            
            # Processamento para outras entidades
            else:
                if label_ent in entidades and texto_ent not in entidades[label_ent]:
                    entidades[label_ent].append(texto_ent)
                    
        # =============================================
        # PÓS-PROCESSAMENTO PARA NOMES DE PACIENTES
        # =============================================
        # 1. Remover nomes que são partes de outros nomes
        nomes_finais = []
        for nome in sorted(entidades["NOME_PACIENTE"], key=len, reverse=True):
            # Verificar se este nome não é parte de um nome mais completo já na lista
            if not any(nome != outro and nome in outro for outro in nomes_finais):
                nomes_finais.append(nome)
        
        # 2. Remover nomes que não atendem aos critérios mínimos
        nomes_validos = [nome for nome in nomes_finais if validar_nome_paciente(nome)]
        
        entidades["NOME_PACIENTE"] = nomes_validos

    return entidades

if __name__ == "__main__":
    exemplos = [
        #01 - Não tem CID
        "ATESTADO Atesto para os devidos fins que FLAVIO AUGUSTO BERNASKI DA SILVA compareceu para atendimento psicológico no dia 05/04/2023 das 11h30 às 12:00.",
        #02 - Não tem CID
        "DECLARAÇÃO Declaro para os devidos fins que o(a) Sr.(a), VITÓRIA BRENDA LUSTOSA SILVA, compareceu a este serviço, como acompanhante do paciente, YARA LUSTOSA DOS SANTOS, no período de 22/05/2024. Maringá, 22 de maio de 2024. Dra. Gina Bressan Schiavon Masson (CRM 21548)",
        #03 - Não tem CID
        "ATESTADO Atesto para os devidos fins que ANDERSON BISPO DOS SANTOS, permaneceu no hospital das 07h43 às 11:35 para consulta médica e realização de exames. Belo Horizonte, Quarta-Feira, 01 de Fevereiro de 2023 CRM-MG - 85985",
        #04 - Tem CID
        "ATESTADO Atesto para os devidos fins que MARIANA SILVA SAULO, portadora do CPF 123.456.789-10, esteve sob meus cuidados médicos nesta unidade de saúde em 07/07/2025. No período das 10:15 às 11h45. Após avaliação clínica, foi diagnosticada com Dor lombar baixa (CID: M54.5). Recomenda-se o afastamento de suas atividades laborais por 3 (três) dias, a partir desta data, para recuperação e repouso.\nPalmares, 7 de julho de 2025. Dr. Ricardo Alves Pereira\nCRM-PE 123456",
        #05 - Não tem CID
        "ATESTADO Atesto que o sr(a) PAULO ANDRÉ FERNANDES compareceu para atendimento psicológico no dia 15/10/2025 das 08h30 às 11:00.",
        #06 - a partir daqui todos tem CID
        "ATESTADO MÉDICO Declaro que o paciente CARLOS EDUARDO MENDONÇA esteve em consulta no dia 12/08/2024 das 14:20 às 15:30. Diagnosticado com pneumonia (CID J18.9). Recomendado repouso por 7 dias. CRM: 34876.",
        #07
        "DECLARAÇÃO Atesto para fins legais que MARIA FERNANDA OLIVEIRA compareceu para tratamento de enxaqueca crônica (CID G43.909) em 30/09/2024 no horário das 09 00 às 10 15. Dr. Roberto Silva - CRM/SP 56789.",
        #08
        "ATESTADO Confirmo que LUCAS RODRIGUES SANTOS foi atendido em nossa clínica em 15/11/2024 entre 08 45 e 10 30. Diagnosticado com diabetes mellitus tipo 2 (CID E11.9). Necessário afastamento por 10 dias. CRM-MG 12345.",
        #09
        "RELATÓRIO MÉDICO Declaro que ANA BEATRIZ COSTA esteve sob meus cuidados em 03/12/2024 das 13 15 às 14 45. Diagnosticada com transtorno de ansiedade generalizada (CID F41.1). Indicação de acompanhamento psicológico semanal. CRM 98765.",
        #10
        "ATESTADO Atesto que RAFAEL PEREIRA LIMA compareceu para consulta de rotina no dia 22/01/2025 das 11 00 às 12 20. Diagnosticado com hipertensão essencial (CID I10). Recomendado controle periódico. CRM/RS 54321.",
        #11
        "DECLARAÇÃO Confirmo que JULIANA SOUZA ALMEIDA foi atendida em 05/02/2025 no período das 16 30 às 17 45. Diagnosticada com asma brônquica não alérgica (CID J45.909). Necessário uso contínuo de medicação. CRM 23456.",
        #12
        "ATESTADO MÉDICO Declaro que PEDRO HENRIQUE BARBOSA esteve em tratamento no dia 18/03/2025 das 10 00 às 11 15. Diagnosticado com artrite reumatoide (CID M06.9). Afastamento recomendado por 15 dias. CRM 87654.",
        #13
        "RELATÓRIO Confirmo que FERNANDA LIMA COSTA compareceu para avaliação em 09/04/2025 no horário das 14 00 às 15 30. Diagnosticada com depressão recorrente (CID F33.9). Indicado tratamento psicoterapêutico. CRM/SP 34567.",
        #14
        "ATESTADO Atesto que GUSTAVO OLIVEIRA SANTOS foi atendido em nossa unidade em 25/05/2025 das 09 30 às 11 00. Diagnosticado com gastrite aguda (CID K29.0). Recomendado dieta e repouso por 5 dias. CRM 76543.",
        #15
        "DECLARAÇÃO MÉDICA Declaro que PATRÍCIA RIBEIRO MARTINS esteve sob cuidados em 07/06/2025 no período das 15 45 às 17 10. Diagnosticada com hipotireoidismo (CID E03.9). Necessário acompanhamento endocrinológico. CRM 45678.",
        #16
        "DECLARAÇÃO Atesto para os devidos fins que PAULO AUGUSTO PEREIRA compareceu para atendimento psicológico no dia 05/05/2024 das 11:30 às 12:00, e precisará se afastar por 30 dias, devido ao CID:M54.5",
        #17
        "DECLARAÇÃO Declaro que LUCAS OLIVEIRA ROCHA esteve em atendimento nesta clínica no dia 15 de setembro de 2025, durante o período da tarde (entre 14:00 e 15:30). Diagnosticado com Sinusite aguda (CID- J01.9). Recomendado afastamento das atividades por 1 dia. CRM/SP 87654.",
        #18
        "RELATÓRIO MÉDICO Consta que FERNANDA COSTA LIMA compareceu à emergência em 30/11/2024 às 22:40, permanecendo até 00:15 (período da noite). Diagnóstico: Gastroenterite (CID_ A09). Necessário afastamento por (5 dias). CRM 44567/PR.",
        #19
        "ATESTADO Atesto que RAFAEL SANTOS DIAS (CPF 987.654.321-00) foi avaliado em 3 de Abril de 2025 das 08:00 às 09:20. Apresenta Quadro de Amigdalite (CID: J03.9). Indica-se repouso domiciliar por um dia. Belo Horizonte, 3 de abril de 2025. Dra. Juliana Moraes | CRM-MG 11223.",
        #20
        "RECEITUÁRIO MÉDICO Confirmo que PATRÍCIA NUNES FONTES realizou consulta em 20/10/2025 no período da manhã (07:30 às 08:45). Diagnóstico: Transtorno de ansiedade (cid- F41.1). Requer afastamento laboral por 15 dias para tratamento. CRM-RJ 55443.",
        #21
        "ATESTADO Declaro que GUSTAVO HENRIQUE MARTINS esteve sob cuidados médicos em 05/Jan/2025 entre 16:10 e 17:30. Diagnosticado com Lombociatalgia (CID_ M54.4). Recomenda-se afastamento por 8 dias, conforme protocolo clínico. Dr. Tiago Albuquerque | CRM: 99876/BA.",
        #22
        "RELATÓRIO Confirmo que SOFIA RAIMUNDA COSTA (CPF 789.012.345-67) foi atendida em 15-Ago-2025 das 07:00 às 08:30. Diagnosticada com Enxaqueca (CiD- G43.909). Recomendado repouso por no período da manhã. CRM/RS 87654.",
        #23
        "ATESTADO Atesto que LUCAS GABRIEL PEIXOTO esteve em consulta em 30 de setembro de 2025. Durante o período das 18:20 às 19:45 (noite). Diagnóstico: Conjuntivite bacteriana (cid: H10.2). Afastamento necessário: (3 dias). Dra. Fernanda Lima | CRM: 11223/SC.",
        #24 não identificou o padrão Período: 13:15-14:45 (tarde)
        "DECLARAÇÃO MÉDICA Consta que MARIANA FONTES RIBEIRO compareceu à unidade em 12/11/2024. Períod: 13:15 as 14:45 (tarde). Diagnóstico: Asma exacerbada (Cid_ J45.901). Repouso indicado por 2 dias. CRM 44556/GO.",
        #25 - não localizou o NOME PACIENTE
        "DECLARAÇÃO MÉDICA Declaro que CARLOS ROBERTO MARTINS esteve sob cuidados em 05-Jan-2026 das 09:00 às 10:30. Diagnosticado com Hipertensão essencial (CID- I10). Afastamento de 15 dias necessário. CRM/MT 33445.",
        #26 - não identificou o horario neste padr~çao (14:00-15:20)
        "ATESTADO MÉDICO Atesto para os devidos fins que JULIANA SANTOS OLIVEIRA, portadora do CPF 234.567.890-11, esteve em atendimento no dia 20 de Fevereiro de 2025 (14:00-15:20). Diagnóstico: Depressão moderada (CID_ F32.1). Repouso por entre 1 a 15 dias. Dr. Roberto Mendes\nCRM-PA 77889.",
        #27 - identificou -> período da manhã, mas não o padrão de horário (08:40-10:00)
        "RELATÓRIO Confirmo atendimento a RAFAEL CARVALHO DIAS em 03/12/2024 no período da manhã (08:40-10:00). Diagnóstico: Gastrite aguda (cid- K29.0). Recomendado afastamento por 1 dia. CRM 99001/AL.",
        #28
        "DECLARACAO Declaro que PATRÍCIA NUNES FERNANDES esteve em consulta em 15-Jul-2025 das 16:00 às 17:30. Diagnosticada com Lombalgia (CID: M54.5). Necessário repouso por no período da tarde. CRM/RN 22334.",
        #29 naão reconheceu o padrão (21:15-22:40)
        "RECEITUÁRIO MÉDICO Consta que ANDRÉ LUIZ ROCHA foi avaliado em 10 de Março de 2025 no hoário de 19:00 as 20:30. Diagnóstico: Insônia (Cid: G47.0). Afastamento indicado: 8 dias. Dra. Camila Porto | CRM-ES 55667.",
        #30 naão reconheceu o padrão (21:15-22:40)
        "ATESTADO Atesto que GUSTAVO HENRIQUE MENDONÇA esteve sob cuidados em 25/04/2025 no período da noite, sendo este entre 21:15 as 22:40. Diagnóstico: Bronquite aguda (CiD_ J20.9). Repouso por 1 (um) dia. CRM 77881/CE.",
        #31
        "RELATÓRIO MÉDICO Declaro atendimento a AMANDA COSTA SILVEIRA em 07-Set-2025 das 11:00 às 12:30. Diagnóstico: Sinusite crônica (CID J32.9). Afastamento necessário: (10 dias). CRM/DF 11223.",
        #32 - reconheceu periodo da manhã apenas, antes (período da tarde)
        "DECLARAÇÃO Confirmo que RODRIGO PEREIRA ALMEIDA (CPF 345.678.901-23) compareceu em 15/06/2025 no período da tarde. Diagnóstico: Tendinite de punho (cid- M65.4). Repouso indicado por no período da manhã. CRM 44556/BA.",
        #33 - não localizou o tipo de documento - FERNANDA LOPES SANTOS
        "ATESTADO Atesto para os devidos fins que Fernanda Lopes Santos esteve em consulta em 30-Nov-2025 das 08:45 às 10:15. Diagnosticada com Ansiedade generalizada (CID_ F41.1). Afastamento de 15 dias. Dra. Isabela Martins\nCRM-MG 66778.",
        #34 - não localizou o nome paciente - nem horario atendimento antes era (16:30-18:00)
        "ATESTADO MÉDICO Atesdo para os devidos fins que o sr(a). MARCOS VINICIUS OLIVEIRA em 22 de Agosto de 2025 das 16:30 as 18:00. Diagnóstico: Artrose de joelho (CID M17). Repouso necessário por 8 dias. CRM/SP 88990.",
        #35
        "RELATÓRIO Consta que a sr(a). BEATRIZ RIBEIRO COSTA foi avaliada em 05/01/2026 no período da manhã (07:30-09:00). Diagnóstico: Anemia ferropriva (cid_ D50.9). Afastamento indicado: entre 1 a 15 dias. CRM 11223/PR.",
        #36
        "DECLARAÇÃO MÉDICA Confirmo atendimento ao sr(a). Anderson Moreira da Costa, em 14-Dez-2025 das 13:00 às 14:45. Diagnóstico: Vertigem (CID H81.9). Necessário repouso por (5 dias). Dr. Felipe Costa | CRM: 33445/SC.",
        #37
        "DECLARAÇÃO MÉDICA Confirmo que DANIEL SOUZA LIMA esteve em 14-Dez-2025 das 13:00 às 14:45. em acompanhamento do seu filho menor de idade. Onde por ventura, necessita de acompanhamento do paciente no período de (5 dias). Dr. Felipe Costa | CRM: 33445/SC."
    ]


    # for i, texto in enumerate(exemplos, start=1):
    #     entidades = extrair_entidades_multimodelo(texto, modelos)
    #     print(f"\n--- Exemplo {i} ---")
    #     print(json.dumps(entidades, indent=4, ensure_ascii=False))
        
    
    resultados = []  # Lista para acumular os resultados

    for i, texto in enumerate(exemplos, start=1):
        entidades = extrair_entidades_multimodelo(texto, modelos)
        resultado = {
            "id": i,
            "texto": texto,
            "entidades": entidades
        }
        resultados.append(resultado)

        # Ainda imprime no console para depuração
        print(f"\n--- Exemplo {i} ---")
        print(json.dumps(entidades, indent=4, ensure_ascii=False))

    # Salvar em JSON no final
    with open("resultados_entidades.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    print("\n✅ Resultados salvos em 'resultados_entidades.json'")
