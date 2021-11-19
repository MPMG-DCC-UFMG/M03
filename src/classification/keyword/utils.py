def get_pages(pdf):
    return pdf.pages

def extract_title(title):
    return ' '.join(re.findall(r"(?=("+'|'.join(keywords)+r"))", title))

def extract_content(content):
    return ' '.join(re.findall(r"(?=("+'|'.join(keywords)+r"))", content))

def title_extraction_bold(pdf_page, words_limit=20):
    title = ""
    for word in pdf_page.extract_words(extra_attrs=["fontname"])[:words_limit]:  # Using [2:] to skip the main title
        if "Bold" in word["fontname"]:
            title = " ".join([title, word["text"]]).strip()
    return title

def title_extraction_upper(pdf_page, words_limit=20):
    title = ""
    for word in pdf_page.extract_words(extra_attrs=["fontname"])[:words_limit]:
        if word["text"].isupper():
            title = " ".join([title, word["text"]]).strip()
    return title

def title_extraction_upper_bold(pdf_page, words_limit=20):
    title = ""
    for word in pdf_page.extract_words(extra_attrs=["fontname"])[:words_limit]:
        if "Bold" in word["fontname"] and word["text"].isupper():
            title = " ".join([title, word["text"]]).strip()
    return title

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

def get_first_tokens(pdf, limit = 1000):
    num_pages = min(len(pdf.pages), 5)
    pages = pdf.pages[:num_pages]
    content = ''
    for page in pages:
        content += page.extract_text()[:limit]
    if bool(content):
        content = text_preprocessing.remove_special_characters(content).lower()
        content = text_preprocessing.remove_excessive_spaces(content)
    else:
        return None
    return content

def table_extraction(pdf, page_number = 0):
    tables = []
    page = pdf.pages[page_number]
    for table in page.extract_tables():
        tables.append(table)
    return tables
