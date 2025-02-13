import pytest
from fastapi.testclient import TestClient
import torch
from unittest.mock import Mock, patch
from src.model.model_server import app, load_model

client = TestClient(app)

@pytest.fixture
def mock_model():
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # Mock tokenizer
            tokenizer = Mock()
            tokenizer.encode.return_value = torch.tensor([1, 2, 3])
            tokenizer.decode.return_value = "Mocked response"
            tokenizer.eos_token_id = 2
            mock_tokenizer.return_value = tokenizer
            
            # Mock model
            model = Mock()
            model.generate.return_value = torch.tensor([[1, 2, 3]])
            model.to.return_value = model
            mock_model.return_value = model
            
            yield model, tokenizer

@pytest.fixture
def sample_generate_request():
    return {
        "prompt": "What is the meaning of life?",
        "context": "Philosophy discusses the meaning of existence.",
        "max_length": 100,
        "temperature": 0.7
    }

@pytest.fixture
def sample_embedding_request():
    return {
        "texts": ["Document 1", "Document 2"]
    }

def test_health_check(mock_model):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate_response(mock_model, sample_generate_request):
    response = client.post("/generate", json=sample_generate_request)
    assert response.status_code == 200
    assert "response" in response.json()
    assert "model_info" in response.json()

def test_generate_no_context(mock_model):
    request = {
        "prompt": "Test prompt",
        "max_length": 100
    }
    response = client.post("/generate", json=request)
    assert response.status_code == 200

def test_embeddings(mock_model, sample_embedding_request):
    response = client.post("/embed", json=sample_embedding_request)
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "dimensions" in response.json()

def test_invalid_generate_request():
    response = client.post("/generate", json={})
    assert response.status_code == 422

def test_model_error_handling(mock_model, sample_generate_request):
    model, _ = mock_model
    model.generate.side_effect = RuntimeError("GPU out of memory")
    response = client.post("/generate", json=sample_generate_request)
    assert response.status_code == 500

@pytest.mark.asyncio
async def test_concurrent_requests(mock_model, sample_generate_request):
    import asyncio
    import httpx
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tasks = []
        for _ in range(5):
            tasks.append(ac.post("/generate", json=sample_generate_request))
        
        responses = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in responses)

def test_long_input_handling(mock_model):
    long_request = {
        "prompt": "x" * 10000,  # Very long input
        "max_length": 100
    }
    response = client.post("/generate", json=long_request)
    assert response.status_code == 200