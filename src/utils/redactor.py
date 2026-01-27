from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import spacy
import logging
from typing import Dict, List

class FinancialRedactor:
    def __init__(self):
        # Use larger spaCy model for better NER
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            logging.warning("Large model not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
            nlp = spacy.load("en_core_web_lg")
        
        self.analyzer = AnalyzerEngine(nlp_engine=nlp)
        self.anonymizer = AnonymizerEngine()
        
        # Add custom patterns for financial documents
        self._add_custom_patterns()
        
        # Track redactions for audit
        self.redaction_log = []
        
    def _add_custom_patterns(self):
        """Add financial-specific patterns"""
        
        # Account numbers (various formats)
        account_pattern = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[
                Pattern("Account Pattern", r"\b\d{8,17}\b", 0.85),
                Pattern("Account Format", r"(?i)(?:account|acct)[\s#:]*(\d{8,17})", 0.9)
            ]
        )
        
        # Routing numbers
        routing_pattern = PatternRecognizer(
            supported_entity="ROUTING_NUMBER", 
            patterns=[Pattern("Routing", r"\b\d{9}\b", 0.7)]
        )
        
        # CEO/Executive names (common ones)
        executive_pattern = PatternRecognizer(
            supported_entity="EXECUTIVE_NAME",
            patterns=[
                Pattern("Common CEOs", 
                       r"(?i)\b(Elon Musk|Jeff Bezos|Tim Cook|Satya Nadella|Mark Zuckerberg)\b", 
                       0.95)
            ]
        )
        
        self.analyzer.registry.add_recognizer(account_pattern)
        self.analyzer.registry.add_recognizer(routing_pattern)
        self.analyzer.registry.add_recognizer(executive_pattern)

    def redact_text(self, text: str, doc_id: str = "unknown") -> Dict[str, any]:
        """
        Redact sensitive information and return both redacted text and metadata
        
        Returns:
            dict: {
                'text': redacted text,
                'redaction_count': number of redactions,
                'entities_found': list of entity types found
            }
        """
        # Analyze for all entity types
        entities_to_detect = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
            "LOCATION", "CREDIT_CARD", "US_SSN", 
            "US_BANK_NUMBER", "IBAN_CODE",
            "ACCOUNT_NUMBER", "ROUTING_NUMBER", "EXECUTIVE_NAME"
        ]
        
        results = self.analyzer.analyze(
            text=text,
            entities=entities_to_detect,
            language='en'
        )
        
        # Log what was found (for audit trail)
        entities_found = list(set([r.entity_type for r in results]))
        
        # Anonymize
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
                "PERSON": OperatorConfig("replace", {"new_value": "[PERSON_NAME]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
                "ACCOUNT_NUMBER": OperatorConfig("replace", {"new_value": "[ACCOUNT_XXX]"}),
            }
        )
        
        # Verify redaction actually happened for known test cases
        redacted_text = anonymized_result.text
        self._verify_redaction(redacted_text)
        
        # Store audit log entry
        log_entry = {
            'doc_id': doc_id,
            'redaction_count': len(results),
            'entities_found': entities_found,
            'original_length': len(text),
            'redacted_length': len(redacted_text)
        }
        self.redaction_log.append(log_entry)
        
        return {
            'text': redacted_text,
            'redaction_count': len(results),
            'entities_found': entities_found
        }
    
    def _verify_redaction(self, text: str):
        """
        Verify that known sensitive patterns were actually redacted
        Raises warning if suspicious content remains
        """
        suspicious_patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'Potential person name'),
            (r'\b\d{3}-\d{2}-\d{4}\b', 'Potential SSN'),
            (r'\b\d{16}\b', 'Potential credit card'),
        ]
        
        for pattern, description in suspicious_patterns:
            import re
            if re.search(pattern, text):
                logging.warning(f"Redaction verification: {description} may still exist in text")
    
    def get_audit_log(self) -> List[Dict]:
        """Return redaction audit log"""
        return self.redaction_log

if __name__ == "__main__":
    redactor = FinancialRedactor()
    
    # Test cases
    test_texts = [
        "Contact John Doe at john.doe@example.com or 555-123-4567",
        "CEO Elon Musk announced...",
        "Account number 1234567890 was debited",
        "SSN: 123-45-6789",
    ]
    
    for text in test_texts:
        result = redactor.redact_text(text)
        print(f"\nOriginal: {text}")
        print(f"Redacted: {result['text']}")
        print(f"Entities: {result['entities_found']}")