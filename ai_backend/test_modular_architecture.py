"""Test cases for modular architecture."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from app.modules.integration import get_container, reset_container


class TestModularArchitecture:
    """Test modular architecture components."""
    
    def setup_method(self):
        """Setup for each test."""
        reset_container()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup after each test."""
        reset_container()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_container_initialization(self):
        """Test container initializes all services."""
        container = get_container()
        
        # Mock settings to use temp directory
        with patch('app.modules.config.settings.DATABASE_DIR', self.temp_dir):
            container.initialize()
        
        # Check all services are initialized
        assert container.get_user_manager() is not None
        assert container.get_session_manager() is not None
        assert container.get_authenticator() is not None
        assert container.get_vector_store() is not None
        assert container.get_embedding_manager() is not None
        assert container.get_document_manager() is not None
        assert container.get_version_manager() is not None
        assert container.get_rag_orchestrator() is not None
    
    @pytest.mark.asyncio
    async def test_user_authentication_flow(self):
        """Test complete user authentication flow."""
        container = get_container()
        
        with patch('app.modules.config.settings.DATABASE_DIR', self.temp_dir):
            container.initialize()
        
        user_manager = container.get_user_manager()
        authenticator = container.get_authenticator()
        
        # Test authentication with default admin user
        user = await authenticator.authenticate("admin", "admin123")
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "SuperAdmin"
        
        # Test token creation and verification
        token = await authenticator.create_access_token(user)
        assert token is not None
        
        verified_user = await authenticator.verify_token(token)
        assert verified_user is not None
        assert verified_user["username"] == "admin"
    
    @pytest.mark.asyncio
    async def test_session_management(self):
        """Test session management functionality."""
        container = get_container()
        
        with patch('app.modules.config.settings.DATABASE_DIR', self.temp_dir):
            container.initialize()
        
        session_manager = container.get_session_manager()
        
        # Create session
        session_id = session_manager.create_session(
            user_id="test_user",
            role="Employee",
            department="Engineering"
        )
        assert session_id is not None
        
        # Get session
        session = session_manager.get_session(session_id)
        assert session is not None
        
        # Store and retrieve messages
        session_manager.store_message(session_id, "user", "Hello")
        session_manager.store_message(session_id, "assistant", "Hi there!")
        
        messages = await session_manager.fetch_recent_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["speaker"] == "user"
        assert messages[0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_document_management(self):
        """Test document management functionality."""
        container = get_container()
        
        with patch('app.modules.config.settings.DATABASE_DIR', self.temp_dir):
            # Mock vector store to avoid ChromaDB dependency
            mock_vector_store = Mock()
            mock_vector_store.add_document = Mock(return_value="doc_123")
            
            container.initialize()
            container.override_instance("vector_store", mock_vector_store)
        
        doc_manager = container.get_document_manager()
        
        # Test document addition
        doc_id = await doc_manager.add_document(
            text="Test document content",
            metadata={"department": "HR", "sensitivity": "public_internal"},
            user={"user_id": "test_user", "role": "Employee"}
        )
        
        assert doc_id is not None
        mock_vector_store.add_document.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_version_management(self):
        """Test version management functionality."""
        container = get_container()
        container.initialize()
        
        version_manager = container.get_version_manager()
        
        # Create version
        version_id = await version_manager.create_version_record(
            document_id="doc_123",
            content="Version 1 content",
            metadata={"author": "test_user"}
        )
        
        assert version_id == "v1"
        
        # Get versions
        versions = version_manager.get_versions("doc_123")
        assert len(versions) == 1
        assert versions[0]["version_id"] == "v1"
        
        # Get specific version
        version = version_manager.get_version("doc_123", "v1")
        assert version is not None
        assert version["content"] == "Version 1 content"
    
    @pytest.mark.asyncio
    async def test_rag_orchestrator(self):
        """Test RAG orchestrator functionality."""
        container = get_container()
        
        with patch('app.modules.config.settings.DATABASE_DIR', self.temp_dir):
            # Mock vector store
            mock_vector_store = Mock()
            mock_vector_store.search_documents = Mock(return_value=[
                {
                    "id": "doc_1",
                    "text": "Test document content",
                    "metadata": {"department": "General", "sensitivity": "public_internal"},
                    "distance": 0.8
                }
            ])
            
            container.initialize()
            container.override_instance("vector_store", mock_vector_store)
        
        rag_orchestrator = container.get_rag_orchestrator()
        
        # Create RAG request
        from app.modules.llm.interfaces import RAGRequest
        request = RAGRequest(
            question="What is the company policy?",
            user={"user_id": "test_user", "role": "Employee", "department": "General"},
            top_k=3,
            use_llm=False  # Skip LLM for testing
        )
        
        # Process query
        response = await rag_orchestrator.process_query(request)
        
        assert response is not None
        assert len(response.retrieved_documents) == 1
        assert response.retrieved_documents[0].text == "Test document content"
        assert response.context is not None
    
    @pytest.mark.asyncio
    async def test_core_utils(self):
        """Test core utility functions."""
        from app.modules.core.utils import analyze_sentiment, truncate_text, extract_keywords
        
        # Test sentiment analysis
        result = analyze_sentiment("I love this product!")
        assert result["sentiment"] == "positive"
        
        result = analyze_sentiment("This is terrible!")
        assert result["sentiment"] == "negative"
        
        result = analyze_sentiment("This is okay.")
        assert result["sentiment"] == "neutral"
        
        # Test text truncation
        text = "This is a long text that needs to be truncated"
        truncated = truncate_text(text, 20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
        
        # Test keyword extraction
        keywords = extract_keywords("The quick brown fox jumps over the lazy dog", max_keywords=3)
        assert len(keywords) <= 3
        assert isinstance(keywords, list)
    
    def test_container_singleton(self):
        """Test container singleton behavior."""
        container1 = get_container()
        container2 = get_container()
        
        assert container1 is container2
        
        # Test reset
        reset_container()
        container3 = get_container()
        assert container3 is not container1
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in modules."""
        container = get_container()
        
        with patch('app.modules.config.settings.DATABASE_DIR', self.temp_dir):
            container.initialize()
        
        authenticator = container.get_authenticator()
        
        # Test invalid authentication
        user = await authenticator.authenticate("invalid", "invalid")
        assert user is None
        
        # Test invalid token verification
        verified = await authenticator.verify_token("invalid_token")
        assert verified is None


def run_tests():
    """Run all tests."""
    print("Running modular architecture tests...")
    
    # Run tests
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()