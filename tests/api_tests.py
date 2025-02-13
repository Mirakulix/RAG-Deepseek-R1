import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
from src.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_chroma():
    with patch('chromadb.HttpClient') as mock:
        collection_mock = Mock()
        mock.return_value.create_collection.return_value = collection_mock
        yield collection_mock

@pytest.fixture
def sample_document():
    return {
        "content": "This is a test document.",
        "metadata": {
            "source": "test",
            "type": "text"
        }
    }

@pytest.fixture
def sample_query():
    return {
        "text": "What is the meaning of life?",
        "context_size": 3
    }

def test_add_document(mock_chroma, sample_document):
    response = client.post("/documents", json=sample_document)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_chroma.add.assert_called_once()

def test_query_documents(mock_chroma, sample_query):
    # Mock ChromaDB response
    mock_chroma.query.return_value = {
        "documents": [["Test document 1", "Test document 2"]],
        "metadatas": [[{"source": "test1"}, {"source": "test2"}]]
    }
    
    # Mock model service response
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "response": "Test response",
            "model_info": {"device": "cuda"}
        }
        
        response = client.post("/query", json=sample_query)
        
        assert response.status_code == 200
        assert "response" in response.json()
        mock_chroma.query.assert_called_once()

def test_invalid_document():
    response = client.post("/documents", json={})
    assert response.status_code == 422

def test_invalid_query():
    response = client.post("/query", json={})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_model_service_error(mock_chroma, sample_query):
    mock_chroma.query.return_value = {
        "documents": [["Test document"]],
        "metadatas": [[{"source": "test"}]]
    }
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.return_value.status_code = 500
        response = client.post("/query", json=sample_query)
        assert response.status_code == 500

def test_large_document_handling(mock_chroma):
    large_doc = {
        "content": "x" * 1000000,  # 1MB document
        "metadata": {"source": "test"}
    }
    response = client.post("/documents", json=large_doc)
    assert response.status_code == 200