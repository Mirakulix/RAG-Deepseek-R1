from typing import Optional, Dict, Any, List
import asyncio
from dataclasses import dataclass
import aiohttp
import logging
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram
import json
import backoff

@dataclass
class ServiceConfig:
    name: str
    host: str
    port: int
    timeout: float
    retry_count: int
    circuit_breaker_threshold: int

class CircuitBreaker:
    def __init__(self, threshold: int = 5, reset_timeout: float = 60.0):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                self.state = "open"
            raise e

class ServiceIntegration:
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)

        # Metrics
        self.request_counter = Counter(
            'service_requests_total',
            'Total number of service requests',
            ['service_name', 'method']
        )
        self.latency_histogram = Histogram(
            'service_request_duration_seconds',
            'Service request duration in seconds',
            ['service_name', 'method']
        )

    async def initialize(self):
        """Initialisiert die Service Integration."""
        self.session = aiohttp.ClientSession()
        
        # Standard-Services registrieren
        self.register_service(ServiceConfig(
            name="model",
            host="deepseek-model-service",
            port=8080,
            timeout=30.0,
            retry_count=3,
            circuit_breaker_threshold=5
        ))
        
        self.register_service(ServiceConfig(
            name="chroma",
            host="chroma-service",
            port=8000,
            timeout=10.0,
            retry_count=3,
            circuit_breaker_threshold=5
        ))

    async def cleanup(self):
        """Cleanup der Service Integration."""
        if self.session:
            await self.session.close()

    def register_service(self, config: ServiceConfig):
        """Registriert einen neuen Service."""
        self.services[config.name] = config
        self.circuit_breakers[config.name] = CircuitBreaker(
            threshold=config.circuit_breaker_threshold
        )

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def call_service(
        self,
        service_name: str,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict:
        """Ruft einen Service mit Retry-Logik und Circuit Breaker auf."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        service_config = self.services[service_name]
        circuit_breaker = self.circuit_breakers[service_name]

        url = f"http://{service_config.host}:{service_config.port}{endpoint}"
        
        async def make_request():
            async with self.latency_histogram.labels(
                service_name,
                method
            ).time():
                self.request_counter.labels(service_name, method).inc()
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    timeout=service_config.timeout
                ) as response:
                    response.raise_for_status()
                    return await response.json()

        try:
            return await circuit_breaker.call(make_request)
        except Exception as e:
            self.logger.error(
                f"Service call failed: {service_name} {method} {endpoint}: {str(e)}"
            )
            raise

class ModelService:
    def __init__(self, service_integration: ServiceIntegration):
        self.service_integration = service_integration

    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generiert eine Antwort vom Modell-Service."""
        return await self.service_integration.call_service(
            service_name="model",
            method="POST",
            endpoint="/generate",
            data={
                "prompt": prompt,
                "context": context
            }
        )

    async def generate_embeddings(
        self,
        texts: List[str]
    ) -> Dict[str, Any]:
        """Generiert Embeddings vom Modell-Service."""
        return await self.service_integration.call_service(
            service_name="model",
            method="POST",
            endpoint="/embed",
            data={
                "texts": texts
            }
        )

class ChromaService:
    def __init__(self, service_integration: ServiceIntegration):
        self.service_integration = service_integration

    async def query_documents(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Führt eine Dokumentenabfrage durch."""
        return await self.service_integration.call_service(
            service_name="chroma",
            method="POST",
            endpoint="/query",
            data={
                "query_text": query_text,
                "n_results": n_results
            }
        )

    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fügt Dokumente hinzu."""
        return await self.service_integration.call_service(
            service_name="chroma",
            method="POST",
            endpoint="/documents",
            data={
                "documents": documents
            }
        )

@asynccontextmanager
async def get_service_integration():
    """Context Manager für Service Integration."""
    service_integration = ServiceIntegration()
    try:
        await service_integration.initialize()
        yield service_integration
    finally:
        await service_integration.cleanup()