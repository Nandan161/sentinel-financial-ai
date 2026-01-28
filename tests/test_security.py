import pytest
import tempfile
import os
from pathlib import Path
from src.utils.redactor import FinancialRedactor
from src.utils.ingestor import FinancialIngestor, DocumentProcessingError
from src.utils.vector_store import FinancialVectorStore, VectorStoreError
from src.engine import FinancialRAGEngine, RAGEngineError

class TestSecurity:
    """Test security measures and fixes"""
    
    def test_redaction_verification(self):
        """Test enhanced redaction verification"""
        redactor = FinancialRedactor()
        
        # Test with known executive names
        test_text = "CEO Elon Musk announced record profits for Tesla"
        result = redactor.redact_text(test_text)
        
        # Verify redaction occurred
        assert "[PERSON_NAME]" in result['text']
        assert "Elon Musk" not in result['text']
        assert result['redaction_count'] > 0
        
        # Test verification function directly
        verification_passed = redactor._verify_redaction(
            result['text'], 
            test_text
        )
        assert verification_passed == True
    
    def test_redaction_bypass_prevention(self):
        """Test that redaction bypass is prevented"""
        redactor = FinancialRedactor()
        
        # Test with all-caps names (common bypass attempt)
        test_text = "CEO ELON MUSK announced record profits"
        result = redactor.redact_text(test_text)
        
        # Should still be redacted
        assert "[PERSON_NAME]" in result['text'] or "ELON MUSK" not in result['text']
    
    def test_filename_sanitization(self):
        """Test filename sanitization prevents path traversal"""
        from app import sanitize_filename
        
        # Test path traversal attempts
        malicious_names = [
            "../../../etc/passwd",
            "file.pdf\\..\\..\\windows",
            "file<>:\"|?*.pdf",
            "file\nwith\nnewlines.pdf"
        ]
        
        for name in malicious_names:
            safe_name = sanitize_filename(name)
            # Should not contain path separators
            assert '/' not in safe_name
            assert '\\' not in safe_name
            assert '..' not in safe_name
            # Should not be empty
            assert safe_name
    
    def test_collection_name_sanitization(self):
        """Test collection name sanitization"""
        from src.utils.vector_store import FinancialVectorStore
        
        store = FinancialVectorStore()
        
        # Test malicious collection names
        malicious_names = [
            "collection; DROP TABLE",
            "collection<script>alert('xss')</script>",
            "123invalid_start",
            "collection with spaces",
            "collection-with-dashes"
        ]
        
        for name in malicious_names:
            safe_name = store._sanitize_collection_name(name)
            # Should be sanitized or empty
            if safe_name:
                assert safe_name.isalnum() or '_' in safe_name
                assert not safe_name[0].isdigit()
    
    def test_input_sanitization(self):
        """Test input sanitization in engine"""
        from src.engine import FinancialRAGEngine
        
        engine = FinancialRAGEngine()
        
        # Test dangerous characters
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE documents; --",
            "query with & | $ special chars",
            "query with ` backticks"
        ]
        
        for input_text in dangerous_inputs:
            safe_text = engine._sanitize_input(input_text)
            # Should remove dangerous characters
            assert '<' not in safe_text
            assert '>' not in safe_text
            assert ';' not in safe_text
            assert '&' not in safe_text
            assert '|' not in safe_text
            assert '$' not in safe_text
            assert '`' not in safe_text
    
    def test_file_upload_validation(self):
        """Test comprehensive file upload validation"""
        from app import validate_upload
        from unittest.mock import Mock
        
        # Test invalid file types
        mock_file = Mock()
        mock_file.name = "document.exe"
        mock_file.size = 1000
        mock_file.read = Mock(return_value=b'%PDF')
        mock_file.seek = Mock()
        
        is_valid, error = validate_upload(mock_file)
        assert not is_valid
        assert "Invalid file type" in error
        
        # Test empty file
        mock_file.name = "empty.pdf"
        mock_file.size = 0
        mock_file.read = Mock(return_value=b'')
        mock_file.seek = Mock()
        
        is_valid, error = validate_upload(mock_file)
        assert not is_valid
        assert "empty" in error.lower()
        
        # Test non-PDF file
        mock_file.name = "fake.pdf"
        mock_file.size = 1000
        mock_file.read = Mock(return_value=b'not_pdf_header')
        mock_file.seek = Mock()
        
        is_valid, error = validate_upload(mock_file)
        assert not is_valid
        assert "not a valid PDF" in error
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling"""
        # Test ingestor error handling
        ingestor = FinancialIngestor()
        
        # Test with non-existent file
        with pytest.raises(DocumentProcessingError):
            ingestor.ingest_document("nonexistent.pdf")
        
        # Test vector store error handling
        store = FinancialVectorStore()
        
        # Test with invalid collection name
        result = store.get_collection_stats("invalid; DROP TABLE")
        assert result['exists'] == False
        
        # Test engine error handling
        engine = FinancialRAGEngine()
        
        # Test with invalid collection names
        result = engine.query("test question", ["invalid; DROP TABLE"])
        assert "Invalid document selection" in result['answer']
        
        # Test with empty question
        result = engine.query("", ["test"])
        assert "Please provide a question" in result['answer']
    
    def test_cache_security(self):
        """Test cache security measures"""
        from src.engine import FinancialRAGEngine
        
        engine = FinancialRAGEngine()
        
        # Test cache size limits
        for i in range(150):  # More than the 100 limit
            engine.query(f"test question {i}", ["test_collection"])
        
        # Cache should not exceed limit
        cache_size = engine.get_cache_size()
        assert cache_size <= 100
    
    def test_audit_logging(self):
        """Test audit logging functionality"""
        redactor = FinancialRedactor()
        
        # Process some text
        test_text = "Contact John Doe at john@example.com"
        result = redactor.redact_text(test_text, doc_id="test_doc")
        
        # Check audit log
        audit_log = redactor.get_audit_log()
        assert len(audit_log) > 0
        assert audit_log[0]['doc_id'] == "test_doc"
        assert audit_log[0]['redaction_count'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])