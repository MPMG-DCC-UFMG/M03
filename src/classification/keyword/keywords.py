
ATA, HOMOLOG, EDITAL, OUTROS = "ATA", "HOMOLOG", "EDITAL", "OUTROS"

keywords = [
    {
        "word": "ata",
        "title_regex": str(r"\bata\b"),
        "content_regex": str(r"\bata\b"),
        "class": ATA,
    },
    {
        "word": "sessão pública",
        "title_regex": str(r"\bsessão pública\b"),
        "content_regex": str(r"\bsessão pública\b"),
        "class": ATA,
    },
    {
        "word": "homolog",
        "title_regex": str(r"\bhomologação\b"),
        "content_regex": str(r"\bhomologação\b"),
        "class": HOMOLOG,
    },
    {
        "word": "adjudicação",
        "title_regex": str(r"\badjudicação\b"),
        "content_regex": str(r"\badjudicação\b"),
        "class": HOMOLOG,
    },
    {
        "word": "convite",
        "title_regex": str(r"\bconvite\b"),
        "content_regex": str(r"\bconvite\b"),
        "class": EDITAL,
    },
    {
        "word": "edital",
        "title_regex": str(r"\bedital\b"),
        "content_regex": str(r"\bedital\b"),
        "class": EDITAL,
    },
    {
        "word": "cronograma",
        "title_regex": str(r"\bcronograma\b"),
        "content_regex": str(r"\bcronograma\b"),
        "class": OUTROS,
    },
    {
        "word": "aditamento",
        "title_regex": str(r"\baditamento\b"),
        "content_regex": str(r"\baditamento\b"),
        "class": OUTROS,
    },
    {
        "word": "retificação",
        "title_regex": str(r"\bretificação\b"),
        "content_regex": str(r"\bretificação\b"),
        "class": OUTROS,
    },
    {
        "word": "contrato administrativo",
        "title_regex": str(r"\bcontrato administrativo\b"),
        "content_regex": str(r"\bcontrato administrativo\b"),
        "class": OUTROS,
    },
    {
        "word": "ordem de serviço",
        "title_regex": str(r"\bordem de serviço\b"),
        "content_regex": str(r"\bordem de serviço\b"),
        "class": OUTROS,
    },
    {
        "word": "resposta",
        "title_regex": str(r"\bresposta\b"),
        "content_regex": str(r"\bresposta\b"),
        "class": OUTROS,
    },
    {
        "word": "extrato",
        "title_regex": str(r"\bextrato\b"),
        "content_regex": str(r"\bextrato\b"),
        "class": OUTROS,
    },
    {
        "word": "diário oficial",
        "title_regex": str(r"\bdiário oficial\b"),
        "content_regex": str(r"\bdiário oficial\b"),
        "class": OUTROS,
    },
    {
        "word": "aviso de",
        "title_regex": str(r"\baviso de\b"),
        "content_regex": str(r"\baviso de\b"),
        "class": OUTROS,
    },
]
