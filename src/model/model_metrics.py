from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Optional, Dict, Any

# Metriken definieren
REQUESTS_TOTAL = Counter(
    'model_requests_total',
    'Total number of requests to the model',
    ['method', 'endpoint']
)

RESPONSE_TIME = Histogram(
    'model_response_time_seconds',
    'Response time in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

TOKEN_USAGE = Counter(
    'model_token_usage_total',
    'Total number of tokens used',
    ['operation']
)

MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Current memory usage of the model'
)

GPU_MEMORY_USAGE = Gauge(
    'model_gpu_memory_usage_bytes',
    'Current GPU memory usage of the model'
)

class MetricsMiddleware:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0

    async def __call__(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        
        # Request metrics
        REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        
        # Latency metrics
        RESPONSE_TIME.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)
        
        return response

def track_token_usage(operation: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Token usage tracking
            if isinstance(result, dict) and 'usage' in result:
                TOKEN_USAGE.labels(
                    operation=operation
                ).inc(result['usage']['total_tokens'])
            
            return result
        return wrapper
    return decorator

class ModelMetricsCollector:
    def __init__(self):
        self.last_collection = 0
        self.collection_interval = 15  # seconds

    def collect_metrics(self):
        current_time = time.time()
        if current_time - self.last_collection < self.collection_interval:
            return
        
        try:
            # Memory metrics
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            MEMORY_USAGE.set(memory_info.rss)
            
            # GPU metrics
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                GPU_MEMORY_USAGE.set(gpu_memory)
                
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
        
        self.last_collection = current_time

class ModelPerformanceMonitor:
    def __init__(self):
        self.metrics_collector = ModelMetricsCollector()
        self.performance_data: Dict[str, Any] = {}

    async def track_inference(self, input_text: str, output_text: str, processing_time: float):
        """Track model inference performance."""
        input_tokens = len(input_text.split())
        output_tokens = len(output_text.split())
        
        self.performance_data.update({
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'processing_time': processing_time,
            'tokens_per_second': (input_tokens + output_tokens) / processing_time
        })
        
        # Update metrics
        self.metrics_collector.collect_metrics()

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            'performance_data': self.performance_data,
            'memory_usage': MEMORY_USAGE._value.get(),
            'gpu_memory_usage': GPU_MEMORY_USAGE._value.get(),
            'total_requests': REQUESTS_TOTAL._value.get(),
            'average_response_time': RESPONSE_TIME._sum.get() / max(RESPONSE_TIME._count.get(), 1)
        }