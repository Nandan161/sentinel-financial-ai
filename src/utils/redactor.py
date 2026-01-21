from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class FinancialRedactor:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact_text(self, text: str):
        # Analyze the text for sensitive entities like names, emails, and phone numbers
        results = self.analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"], language='en')
        
        # Anonymize the detected entities
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )
        return anonymized_result.text

# Quick test logic
if __name__ == "__main__":
    redactor = FinancialRedactor()
    sample = "Contact John Doe at john.doe@natwest.com regarding account in London."
    print(f"Original: {sample}")
    print(f"Redacted: {redactor.redact_text(sample)}")