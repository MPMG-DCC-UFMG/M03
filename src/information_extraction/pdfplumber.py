import glob
import json
import os, os.path
import random
import re
import shutil
import pandas as pd
import pdfplumber
import utils.preprocessing_portuguese as preprossPT
from tqdm.notebook import trange, tqdm
from collections import defaultdict


def get_pages(pdf):
    return pdf.pages


def title_extraction_breaklines(pdf, words_limit=20):
    page = pdf.pages[0]
    content = page.extract_text()
    first_lines = []
    if bool(content):
        content = text_preprocessing.remove_special_characters(content, exceptions=["\n"]).lower()
        content = text_preprocessing.remove_excessive_spaces(content)
        content = re.sub(r"\n\s*\n", "\n", content)
        first_lines = content.split("\n", 6)[:-1]
    else:
        return None
    return first_lines


def get_first_page_content(pdf):
    page = pdf.pages[0]
    content = page.extract_text()
    if bool(content):
        content = text_preprocessing.remove_special_characters(content).lower()
        content = text_preprocessing.remove_excessive_spaces(content)
    else:
        return None
    return content


def get_pages_content(pdf, n):
    full_content = ''
    for i in range(n):
        page = pdf.pages[i]
        content = page.extract_text()
        try:
            content = text_preprocessing.remove_special_characters(content).lower()
            content = text_preprocessing.remove_excessive_spaces(content)
            full_content = full_content + " " + content
        except:
            continue
    return full_content


def get_meta_classe(matches_dict):
    ## se houver uma palavra que "anule" alguma das meta classes (ex: retificação de edital)
    title_counts = dict((k, matches_dict[k]) for k in matches_dict if k in title_keys)
    # if title_counts["outros_title_count"] > 0:
    #     return OUTROS
    # if title_counts["homolog_title_count"] > 0:
    #     return HOMOLOG
    title_counts = [(k, v) for k, v in sorted(title_counts.items(), key=lambda item: item[1], reverse=True)]

    content_counts = dict((k, matches_dict[k]) for k in matches_dict if k in content_keys)
    content_counts = [(k, v)for k, v in sorted(content_counts.items(), key=lambda item: item[1], reverse=True)]

    doc_class = ""
    ## se a palavra chave estiver no título tem um peso maior
    if title_counts[0][1] > 0:
        doc_class = key_to_class(title_counts[0][0])
    elif content_counts[0][1] > 0:
        doc_class = key_to_class(content_counts[0][0])
    else:
        doc_class = OUTROS
    return doc_class


def key_to_class(key):
    return {
        "ata_title_count": ATA,
        "ata_content_count": ATA,
        "homolog_title_count": HOMOLOG,
        "homolog_content_count": HOMOLOG,
        "edital_title_count": EDITAL,
        "edital_content_count": EDITAL,
        "errata_title_count": ERRATA,
        "errata_content_count": ERRATA,
    }.get(key, OUTROS)


def update_class_count(doc_class, ata_count, homolog_count, edital_count, others_count, errata_count):
#     print('doc_class:', doc_class)
    if doc_class == ATA:
        ata_count += 1
    if doc_class == HOMOLOG:
        homolog_count += 1
    if doc_class == EDITAL:
        edital_count += 1
    if doc_class == OUTROS:
        others_count += 1
    if doc_class == ERRATA:
        errata_count += 1
    return ata_count, homolog_count, edital_count, others_count, errata_count


def conditions_of_interest(ata_count, edital_count, homolog_count, others_count,
                           errata_count, classes_of_interest, minimum=20):
    condition = False
    if ATA in classes_of_interest:
        condition = condition or ata_count < minimum
    if EDITAL in classes_of_interest:
        condition = condition or edital_count < minimum
    if HOMOLOG in classes_of_interest:
        condition = condition or homolog_count < minimum
    if OUTROS in classes_of_interest:
        condition = condition or others_count < minimum
    if ERRATA in classes_of_interest:
        condition = condition or errata_count < minimum
    return condition


def get_content_matches(title, content):
    matches_dict = {
        "doc_id": pdf_id, "title": title, "city": city, "all_matches": [],
        "ata_title_matches": [], "ata_content_matches": [], "ata_title_count": 0, "ata_content_count": 0,
        "homolog_title_matches": [], "homolog_content_matches": [], "homolog_title_count": 0, "homolog_content_count": 0,
        "edital_title_matches": [], "edital_content_matches": [], "edital_title_count": 0, "edital_content_count": 0,
        "outros_title_matches": [], "outros_content_matches": [], "outros_title_count": 0, "outros_content_count": 0,
        "errata_title_matches": [], "errata_content_matches": [], "errata_title_count": 0, "errata_content_count": 0,
    }

    for word_dict in keywords:
        word = word_dict["word"]
        title_regex = word_dict["title_regex"]
        content_regex = word_dict["title_regex"]
        doc_class = word_dict["class"].lower()
        title_matches = []

        for index in range(len(title)):
            line = title[index]
            match = re.findall(title_regex, line.lower())

            if bool(match) and len(match) > 0:
                title_matches.append({"match": match, "line": index + 1})
                matches_dict[f"{doc_class}_title_matches"] += title_matches
                matches_dict[f"{doc_class}_title_count"] += len(title_matches)

        content_matches = re.findall(content_regex, content.lower())
        matches_dict[f"{doc_class}_content_matches"] += content_matches
        matches_dict[f"{doc_class}_content_count"] += max(len(content_matches) - len(title_matches), 0)
        matches_dict["all_matches"] += content_matches
    return matches_dict
