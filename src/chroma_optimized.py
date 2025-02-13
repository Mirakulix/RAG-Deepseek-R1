import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from cachetools import TTLCache, LRUCache

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class OptimizedChromaDB:
    def __init__(self, host: str, port: int, collection_name: str):
        self.settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=port,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.HttpClient(settings=self.settings)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        # Caches
        self.embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.query_cache = LRUCache(maxsize=100)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger = logging.getLogger(__name__)

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Optimized for semantic search
            )

    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 100):
        """Add documents to ChromaDB with batching and parallel processing."""
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Parallel embedding generation
                embedding_tasks = []
                for doc in batch:
                    if doc.content in self.embedding_cache:
                        doc.embedding = self.embedding_cache[doc.content]
                    else:
                        embedding_tasks.append(doc)
                
                if embedding_tasks:
                    # Process embeddings in parallel
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        self.executor,
                        self._batch_generate_embeddings,
                        embedding_tasks
                    )
                    
                    # Update cache
                    for doc, embedding in zip(embedding_tasks, embeddings):
                        self.embedding_cache[doc.content] = embedding
                        doc.embedding = embedding
                
                # Prepare batch data
                ids = [f"doc_{i}" for i in range(len(batch))]
                embeddings = [doc.embedding for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                documents = [doc.content for doc in batch]
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                self.logger.info(f"Added batch of {len(batch)} documents")
                
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise

    async def query_documents(
        self,
        query_text: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query documents with caching and metadata filtering."""
        cache_key = f"{query_text}_{n_results}_{str(metadata_filter)}"
        
        # Check cache
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        try:
            # Prepare query
            where = metadata_filter if metadata_filter else {}
            
            # Execute query
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0]
            }
            
            # Update cache
            self.query_cache[cache_key] = processed_results
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error querying documents: {str(e)}")
            raise

    def _batch_generate_embeddings(self, documents: List[DocumentChunk]) -> List[np.ndarray]:
        """Generate embeddings for a batch of documents."""
        try:
            texts = [doc.content for doc in documents]
            embeddings = self.collection._embedding_function(texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def delete_documents(self, ids: List[str]):
        """Delete documents from ChromaDB and cache."""
        try:
            self.collection.delete(ids=ids)
            
            # Clear relevant cache entries
            self.query_cache.clear()  # Clear query cache as results might change
            
            self.logger.info(f"Deleted {len(ids)} documents")
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise

    async def update_documents(self, documents: List[DocumentChunk], ids: List[str]):
        """Update existing documents."""
        try:
            # Delete old documents
            await self.delete_documents(ids)
            
            # Add new documents
            await self.add_documents(documents)
            
        except Exception as e:
            self.logger.error(f"Error updating documents: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            peek = self.collection.peek()
            
            return {
                "document_count": count,
                "sample_documents": peek["documents"],
                "embedding_dimension": len(peek["embeddings"][0]) if peek["embeddings"] else None,
                "cache_stats": {
                    "embedding_cache_size": len(self.embedding_cache),
                    "query_cache_size": len(self.query_cache)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            raise