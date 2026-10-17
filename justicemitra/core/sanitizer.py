"""Privacy sanitization, PII redaction, and emergency detection guardrails."""

import re
from typing import Tuple, List, Dict

# Regex patterns for sensitive identifiers
PHONE_PATTERN = re.compile(
    r'(?:(?:\+91[\-\s]?)?|0)?[6-9]\d{4}[\-\s]?\d{5}\b'
)
AADHAAR_PATTERN = re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

# Distress / Emergency Keywords
EMERGENCY_KEYWORDS = [
    "help me right now",
    "kill myself",
    "suicide",
    "beating me right now",
    "life in danger",
    "threat to life",
    "locked me out",
    "immediate danger",
    "physical violence now",
    "attacked right now",
    "call police",
]

# Standard Emergency Helplines for Women in India
EMERGENCY_HELPLINES: List[Dict[str, str]] = [
    {
        "name": "Women Helpline (All India)",
        "number": "1091",
        "description": "24/7 dedicated helpline for women in distress.",
    },
    {
        "name": "National Emergency Number",
        "number": "112",
        "description": "Integrated emergency response for police, fire, and medical assistance.",
    },
    {
        "name": "National Commission for Women (NCW)",
        "number": "7827170170",
        "description": "24/7 helpline for domestic violence and harassment support.",
    },
    {
        "name": "Domestic Violence Helpline (WCD)",
        "number": "181",
        "description": "Toll-free 24-hour service for women facing domestic abuse.",
    },
    {
        "name": "National Legal Services Authority (NALSA)",
        "number": "1516",
        "description": "Free legal aid and assistance for eligible individuals.",
    },
]


def scrub_pii(text: str) -> str:
    """Removes personal identifiable information (phone numbers, Aadhaar, email) from text.
    
    Returns the sanitized text with standard placeholder tags.
    """
    if not text:
        return ""
    
    # Redact phone numbers
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    # Redact 12-digit Indian Aadhaar patterns
    text = AADHAAR_PATTERN.sub("[REDACTED_AADHAAR]", text)
    # Redact emails
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    
    return text


def detect_emergency(text: str) -> Tuple[bool, List[str]]:
    """Analyzes user query for acute distress or immediate emergency keywords.
    
    Returns a tuple of (is_emergency: bool, matched_keywords: List[str]).
    """
    if not text:
        return False, []
    
    lower_text = text.lower()
    matched = [kw for kw in EMERGENCY_KEYWORDS if kw in lower_text]
    return (len(matched) > 0, matched)
