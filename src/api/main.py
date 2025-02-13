from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import chromadb
from typing import List, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API")

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host=os.getenv("CHROMA_HOST", "chroma"), port=8000)
collection = chroma_client.create_collection("documents")

class Query(BaseModel):
    text: str
    context_size: Optional[int] = 3

class Document(BaseModel):
    content: str
    metadata: dict

@app.post("/query")
async def query_documents(query: Query):
    try:
        # Get relevant documents from ChromaDB
        results = collection.query(
            query_texts=[query.text],
            n_results=query.context_size
        )
        
        # Prepare context from retrieved documents
        context = "\n".join([doc for doc in results['documents'][0]])
        
        # Query the DeepSeek model
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{os.getenv('MODEL_SERVICE_URL')}/generate",
                json={
                    "prompt": query.text,
                    "context": context
                }
            )
            
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Model service error")
            
        return response.json()
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_document(document: Document):
    try:
        # Add document to ChromaDB
        collection.add(
            documents=[document.content],
            metadatas=[document.metadata],
            ids=[f"doc_{len(collection.get()['ids']) + 1}"]
        )
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))