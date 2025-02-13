from typing import Optional, Dict, Any, Callable
import functools
import asyncio
import logging
import traceback
from datetime import datetime
from dataclasses import dataclass
import json

@dataclass
class ErrorContext:
    error_type: str
    message: str
    timestamp: datetime
    trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
    def register_handler(self, error_type: str, handler: Callable):
        """Register an error handler for a specific error type."""
        self.error_handlers[error_type] = handler
        
    def register_recovery(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy

def with_error_handling(max_retries: int = 3, retry_delay: float = 1.0):
    """Decorator for error handling and automatic retry."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    last_error = e
                    
                    error_context = ErrorContext(
                        error_type=type(e).__name__,
                        message=str(e),
                        timestamp=datetime.now(),
                        trace=traceback.format_exc(),
                        metadata={
                            'function': func.__name__,
                            'args': str(args),
                            'kwargs': str(kwargs),
                            'retry_count': retries
                        }
                    )
                    
                    # Log error
                    logging.error(f"Error in {func.__name__}: {str(e)}")
                    logging.debug(f"Error context: {error_context}")
                    
                    # Apply recovery strategy if available
                    handler = error_handler.error_handlers.get(type(e).__name__)
                    if handler:
                        try:
                            await handler(error_context)
                        except Exception as recovery_error:
                            logging.error(f"Recovery failed: {str(recovery_error)}")
                    
                    if retries < max_retries:
                        await asyncio.sleep(retry_delay * (2 ** (retries - 1)))  # Exponential backoff
                    
            # All retries failed
            raise last_error
            
        return wrapper
    return decorator

class ModelError(Exception):
    """Base class for model-related errors."""
    pass

class EmbeddingError(ModelError):
    """Error during embedding generation."""
    pass

class TokenLimitError(ModelError):
    """Error when token limit is exceeded."""
    pass

# Error handlers
async def handle_model_error(context: ErrorContext):
    """Handle model-related errors."""
    try:
        # Log to monitoring system
        logging.error(f"Model error: {context.message}")
        
        # Notify administrators
        await notify_admins(context)
        
        # Update metrics
        update_error_metrics(context)
        
    except Exception as e:
        logging.error(f"Error handler failed: {str(e)}")

async def handle_embedding_error(context: ErrorContext):
    """Handle embedding generation errors."""
    try:
        # Clear problematic cache entries
        await clear_embedding_cache()
        
        # Try alternative embedding model
        await switch_to_backup_embedding_model()
        
    except Exception as e:
        logging.error(f"Embedding error handler failed: {str(e)}")

async def handle_token_limit_error(context: ErrorContext):
    """Handle token limit exceeded errors."""
    try:
        # Reduce chunk size
        await adjust_chunk_size(reduction_factor=0.8)
        
        # Clear affected caches
        await clear_document_cache()
        
    except Exception as e:
        logging.error(f"Token limit error handler failed: {str(e)}")

# Recovery strategies
async def basic_retry_strategy(context: ErrorContext):
    """Basic retry strategy with exponential backoff."""
    retry_count = context.metadata.get('retry_count', 0)
    await asyncio.sleep(2 ** retry_count)

async def fallback_model_strategy(context: ErrorContext):
    """Switch to fallback model."""
    try:
        await switch_to_fallback_model()
    except Exception as e:
        logging.error(f"Fallback model strategy failed: {str(e)}")

async def circuit_breaker_strategy(context: ErrorContext):
    """Implement circuit breaker pattern."""
    try:
        await activate_circuit_breaker()
    except Exception as e:
        logging.error(f"Circuit breaker strategy failed: {str(e)}")

# Initialize error handler
error_handler = ErrorHandler()

# Register handlers
error_handler.register_handler("ModelError", handle_model_error)
error_handler.register_handler("EmbeddingError", handle_embedding_error)
error_handler.register_handler("TokenLimitError", handle_token_limit_error)

# Register recovery strategies
error_handler.register_recovery("ModelError", fallback_model_strategy)
error_handler.register_recovery("EmbeddingError", basic_retry_strategy)
error_handler.register_recovery("TokenLimitError", circuit_breaker_strategy)