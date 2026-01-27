import pytest
from src.utils.redactor import FinancialRedactor

def test_person_redaction():
    redactor = FinancialRedactor()
    text = "Contact John Doe at john@example.com"
    result = redactor.redact_text(text)
    
    assert "[PERSON" in result['text']
    assert "[EMAIL]" in result['text']
    assert "John Doe" not in result['text']
    assert result['redaction_count'] >= 2

def test_ceo_names():
    redactor = FinancialRedactor()
    text = "CEO Elon Musk announced record profits"
    result = redactor.redact_text(text)
    
    assert "Elon Musk" not in result['text']
    assert result['redaction_count'] >= 1

def test_account_numbers():
    redactor = FinancialRedactor()
    text = "Account 12345678 was charged"
    result = redactor.redact_text(text)
    
    assert "12345678" not in result['text']