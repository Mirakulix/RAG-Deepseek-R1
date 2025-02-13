import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="DeepSeek-R1 Model Server")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/models/deepseek-r1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "2048"))
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

class GenerateRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    max_length: Optional[int] = MAX_LENGTH
    temperature: Optional[float] = TEMPERATURE

class EmbeddingRequest(BaseModel):
    texts: List[str]

def load_model():
    """Load the DeepSeek-R1 model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Quantization for memory efficiency
        )
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load model on startup
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model, tokenizer = load_model()
    logger.info("Model loaded successfully")

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        # Prepare input text
        input_text = request.prompt
        if request.context:
            input_text = f"Context: {request.context}\n\nQuestion: {request.prompt}\n\nAnswer:"

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = inputs.to(DEVICE)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "response": response,
            "input_text": input_text,
            "model_info": {
                "device": str(DEVICE),
                "max_length": request.max_length,
                "temperature": request.temperature
            }
        }

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def get_embeddings(request: EmbeddingRequest):
    try:
        # Get embeddings using the same model
        with torch.no_grad():
            inputs = tokenizer(
                request.texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Get the hidden states from the last layer
            outputs = model(**inputs, output_hidden_states=True)
            
            # Use mean pooling of last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs['attention_mask']
            
            # Calculate mean pooling
            embeddings = []
            for i in range(len(request.texts)):
                mask = attention_mask[i].unsqueeze(-1)
                masked_embeddings = last_hidden_state[i] * mask
                mean_pooled = torch.sum(masked_embeddings, dim=0) / torch.sum(mask)
                embeddings.append(mean_pooled.cpu().numpy())
            
            embeddings = np.array(embeddings)
            
            return {
                "embeddings": embeddings.tolist(),
                "dimensions": embeddings.shape[-1]
            }

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(DEVICE)}