import re
from typing import List


ABBREV_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")


def tokenize_ko(text: str) -> List[str]:
    text = re.sub(r"[\t\n\r]+", " ", text)
    text = re.sub(r"[^0-9A-Za-z가-힣%/·\-\. ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split(" ") if text else []
    return tokens


def has_abbreviation(text: str) -> bool:
    # Heuristic: consecutive uppercase letters or common medical uppercase abbreviations
    if re.search(ABBREV_PATTERN, text or ""):
        return True
    # Mixed Latin+digits like "HBV", "H1N1" etc.
    return bool(re.search(r"\b[A-Z][A-Z0-9]{1,5}\b", text or ""))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
