"""Unit tests for PII sanitization and emergency detection guardrails."""

import pytest
from justicemitra.core.sanitizer import scrub_pii, detect_emergency, EMERGENCY_HELPLINES


def test_scrub_pii_phone_numbers():
    text = "My phone is 9876543210 and my brother's number is +91-9123456789."
    sanitized = scrub_pii(text)
    assert "9876543210" not in sanitized
    assert "9123456789" not in sanitized
    assert "[REDACTED_PHONE]" in sanitized


def test_scrub_pii_aadhaar():
    text = "My Aadhaar card number is 1234 5678 9012 for verification."
    sanitized = scrub_pii(text)
    assert "1234 5678 9012" not in sanitized
    assert "[REDACTED_AADHAAR]" in sanitized


def test_scrub_pii_email():
    text = "Contact me at victim.support@example.org immediately."
    sanitized = scrub_pii(text)
    assert "victim.support@example.org" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized


def test_detect_emergency_acute_distress():
    text = "My husband has locked me out of the house and I am in immediate danger, please help me right now"
    is_emergency, matches = detect_emergency(text)
    assert is_emergency is True
    assert len(matches) > 0


def test_detect_emergency_standard_inquiry():
    text = "What are the maintenance rights of a married woman under Hindu Marriage Act?"
    is_emergency, matches = detect_emergency(text)
    assert is_emergency is False
    assert len(matches) == 0


def test_emergency_helplines_present():
    assert len(EMERGENCY_HELPLINES) >= 5
    numbers = [h["number"] for h in EMERGENCY_HELPLINES]
    assert "1091" in numbers  # Women Helpline
    assert "112" in numbers   # National Emergency
    assert "181" in numbers   # Domestic Violence
    assert "1516" in numbers  # NALSA Free Legal Aid
