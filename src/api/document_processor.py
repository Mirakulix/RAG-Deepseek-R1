from typing import List, Dict, Any, Optional
import hashlib
import re
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import nltk
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from markdown import markdown
import html2text

@dataclass
class ProcessedChunk:
    content: str
    metadata: Dict[str, Any]
    hash: str
    quality_score: float

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            self.logger.warning(f"NLTK download failed: {str(e)}")

    async def process_document(self, content: str, metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """Process a document into optimized chunks."""
        try:
            # Clean and normalize content
            cleaned_content = await self._clean_content(content, metadata.get('type', 'text'))
            
            # Split into chunks
            chunks = await self._create_chunks(cleaned_content)
            
            # Process chunks in parallel
            loop = asyncio.get_event_loop()
            processed_chunks = await loop.run_in_executor(
                self.executor,
                self._process_chunks,
                chunks,
                metadata
            )
            
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    async def _clean_content(self, content: str, doc_type: str) -> str:
        """Clean and normalize document content based on type."""
        try:
            if doc_type == 'html':
                # Convert HTML to markdown and then to plain text
                html_converter = html2text.HTML2Text()
                html_converter.ignore_links = True
                content = html_converter.handle(content)
            elif doc_type == 'pdf':
                # Extract text from PDF
                doc = fitz.open("pdf", content.encode())
                content = " ".join(page.get_text() for page in doc)
            elif doc_type == 'markdown':
                # Convert markdown to plain text
                content = html2text.html2text(markdown(content))
            
            # General cleaning
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = re.sub(r'[^\w\s.,!?-]', '', content)  # Remove special characters
            content = content.strip()
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error cleaning content: {str(e)}")
            raise

    async def _create_chunks(self, content: str) -> List[str]:
        """Split content into overlapping chunks."""
        try:
            sentences = nltk.sent_tokenize(content)
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence.split())
                
                if current_size + sentence_size > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + [sentence]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Add final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            raise

    def _process_chunks(self, chunks: List[str], metadata: Dict[str, Any]) -> List[ProcessedChunk]:
        """Process chunks in parallel."""
        try:
            processed_chunks = []
            
            for chunk in chunks:
                # Calculate chunk hash
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(chunk)
                
                # Update chunk metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_hash': chunk_hash,
                    'chunk_size': len(chunk.split()),
                    'quality_score': quality_score
                })
                
                processed_chunks.append(ProcessedChunk(
                    content=chunk,
                    metadata=chunk_metadata,
                    hash=chunk_hash,
                    quality_score=quality_score
                ))
            
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing chunks: {str(e)}")
            raise

    def _calculate_quality_score(self, chunk: str) -> float:
        """Calculate a quality score for the chunk."""
        try:
            # Factors for quality score
            factors = {
                'length': 0.3,  # Optimal length
                'coherence': 0.3,  # Sentence coherence
                'info_density': 0.4  # Information density
            }
            
            scores = {}
            
            # Length score
            chunk_length = len(chunk.split())
            scores['length'] = min(chunk_length / self.chunk_size, 1.0)
            
            # Coherence score
            sentences = nltk.sent_tokenize(chunk)
            scores['coherence'] = min(len(sentences) / 5, 1.0)
            
            # Information density score
            pos_tags = nltk.pos_tag(nltk.word_tokenize(chunk))
            content_words = len([word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ', 'RB'))])
            scores['info_density'] = content_words / max(chunk_length, 1)
            
            # Calculate weighted score
            total_score = sum(score * weight for (metric, score), (_, weight) 
                            in zip(scores.items(), factors.items()))
            
            return round(total_score, 3)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0