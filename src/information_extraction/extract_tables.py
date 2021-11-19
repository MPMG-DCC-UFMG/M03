import pdfplumber
import os
import json
import re
from collections import defaultdict
import tabula
import pandas as pd
import camelot
import tabula
import math
import traceback


def get_atas(cities_names):

    caracterizacao_xls =  pd.ExcelFile('../data/resultados_classificacao/resultado_parcial.xlsx')
    ata_paths = {}
    for city, folders in cities_names.items():
        aux_df = pd.read_excel(caracterizacao_xls, city)
        for folder in folders:
            ata_paths[city] = list(map(lambda id:  f'{folder}/{id}.pdf', list(aux_df[aux_df['real_meta-class'] == 'ATA']['doc_id'])))
    return ata_paths


# escolhe a tabela com menos células vazias (com "" ou None)
def select_tables_from_strategy(result_tables):
    count_empty = math.inf
    selected_tables = result_tables[0]
    for tables in result_tables:
        count = count_table_empty(tables['tables'])
        if count < count_empty:
            count_empty = count
            selected_tables = tables
    return selected_tables


# conta a quantidade de células vazias de uma tabela
def count_table_empty(tables):
    cell_count = 0
    empty_count = 0
    for table in tables:
        for line in table:
            empty_count += line.count(None) + line.count('')
            cell_count += len(line)
    frac_empty = empty_count/cell_count if empty_count > 0 and cell_count > 0 else 0
    return frac_empty


# conta a quantidade de células vazias de uma linha
def count_line_empty(line):
    return line.count(None) + line.count('')


# extrai tabelas da página p usando a estégia definida
def pdfplumber_extract_page(pdf, p, settings = {'snap_tolerance': 7, "horizontal_strategy": "text"}):
    page = pdf.pages[p]
    p_tables = pdf.pages[0].extract_tables()
    p_tables = pdf.pages[p].extract_tables(settings)
    return {'page': p, 'tables': p_tables}


# retorna o id de um documento a partir do caminho
def extract_id_from_path(file_path):
    id = file_path.split('/')[-1].split('_')[0]
    return id


def remove_none(line):
    new_line = []
    for content in line:
        if content != None:
            new_line.append(content)
    return new_line


# ignora todas as células None e armazena o índice de todas que sejam vazias ("")
def process_empty(line):
    processed_line = []
    empty_indexes = []
    for index in range(len(line)):
        content = line[index]
        if content != None:
            processed_line.append(content)
        if content == '':
            empty_indexes.append(index)
    return processed_line, empty_indexes


def process_line(index, lines, previous_line):
    processed_info = process_empty(lines[index])
    processed_line, empty_indexes = processed_info[0], processed_info[1]
    processed_line_empty_cells = count_line_empty(processed_line)

    # se não é a última linha
    if index < len(lines) - 1:
        # utiliza como referência 3 linhas abaixo
        reference_lines = [remove_none(line) for line in lines[index + 1:min(len(lines),index + 3)]]
    else:
        # utiliza como referência 3 linhas acima
        reference_lines = [remove_none(line) for line in lines[max(0,index - 3):index]]

    # se não existirem linhas de referência
    if len(reference_lines) == 0:
        reference_lines_size = 0
        reference_empty_lines = 0
    else: # quantidade média de células totais e vazias das linhas de referência
        reference_lines_size = int(sum([len(line) for line in reference_lines])/len(reference_lines))
        reference_empty_cells = sum([count_line_empty(line) for line in reference_lines])/len(reference_lines)

    # se o tamanho da linha atual for diferente das linhas de referência
    if len(processed_line) != reference_lines_size:
        new_line = [processed_line[index] for index in range(len(processed_line)) if index not in empty_indexes]
        # testa remover células vazias ("") e verifica se o tamanho coincide
        # se coincidir, pode ser o caso de as células vazias serem erro do extrator
        if len(new_line) == reference_lines_size:
            processed_line = new_line

    # se a quantidade de células no total for a mesma da linha anterior mas a quantidade de células vazias for pelo menos 50% maior
    elif  bool(previous_line) and len(previous_line) == len(processed_line) and processed_line_empty_cells != reference_empty_cells and (reference_empty_cells/len(processed_line)) >= 0.5:
            processed_line = [f'{previous_line[i]} {processed_line[i]}' for i in range(len(previous_line))]
            processed_line.append(REMOVAL_HASH)
    return processed_line


def extract_file_tables(file):
    tables_extracted = []
    if bool(file) and len(file) > 0:
        processed_table = []
        previous_page = file[0]['page']
        previous_line = None # linha anterior que será usada como referência
        for i in range(len(file)): # para cada página do documento
            tables = file[i]['tables']
            current_page = file[i]['page']
            for j in range(len(tables)): # para cada tabela da página
                lines = tables[j]
                for k in range(len(lines)): # para cada linha da tabela
                    processed_line = process_line(k, lines, previous_line)
                    # se o hash existir significa que a anterior pode ser sobrescrita
                    if REMOVAL_HASH in processed_line:
                        processed_line.remove(REMOVAL_HASH)
                        processed_table[-1] = processed_line
                    # se o tamanho da linha for diferente da processada anteriormente, indica o início de uma nova tabela
                    elif len(processed_table) > 0 and len(processed_line) != len(processed_table[-1]) and count_line_empty(processed_line) != len(processed_line):
                        if len(processed_table[0]) > 0:
                            tables_extracted.append(processed_table)
                        processed_table = []
                        processed_table.append(processed_line)
                    else: # se for igual, a linha é adicionada a tabela
                        processed_table.append(processed_line)
                    previous_line = processed_line
                previous_page = current_page
        # para inserir a última tabela do documento
        tables_extracted.append(processed_table)
    return tables_extracted


# útil para analisar documentos específicos
def filter_tables_by_file_id(file_id):
    return {k: {file_id: v[file_id]} for k, v in cities_tables.items() if file_id in v.keys()}
