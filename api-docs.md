# RAG System API Dokumentation

## Übersicht

Die RAG System API bietet Endpunkte für Dokumentenverarbeitung, Abfragen und Embedding-Generierung. Alle Endpunkte unterstützen JSON und verwenden Bearer Token Authentication.

## Base URL

```
http://your-domain.com/api/v1
```

## Authentication

```http
Authorization: Bearer <your_token>
```

## Endpunkte

### Dokumente

#### POST /documents
Fügt ein neues Dokument zur Wissensbasis hinzu.

**Request Body:**
```json
{
  "content": "string",
  "metadata": {
    "source": "string",
    "type": "string",
    "timestamp": "string",
    "tags": ["string"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "document_id": "string"
}
```

#### GET /documents/{document_id}
Ruft ein spezifisches Dokument ab.

**Response:**
```json
{
  "content": "string",
  "metadata": {
    "source": "string",
    "type": "string",
    "timestamp": "string",
    "tags": ["string"]
  },
  "embeddings": {
    "status": "generated",
    "timestamp": "string"
  }
}
```

### Abfragen

#### POST /query
Führt eine RAG-basierte Abfrage durch.

**Request Body:**
```json
{
  "text": "string",
  "context_size": integer (optional, default: 3),
  "max_length": integer (optional, default: 2048),
  "temperature": float (optional, default: 0.7)
}
```

**Response:**
```json
{
  "response": "string",
  "context": ["string"],
  "metadata": {
    "tokens_used": integer,
    "processing_time": float,
    "source_documents": ["string"]
  }
}
```

### Embeddings

#### POST /embed
Generiert Embeddings für gegebene Texte.

**Request Body:**
```json
{
  "texts": ["string"]
}
```

**Response:**
```json
{
  "embeddings": [[float]],
  "dimensions": integer
}
```

### System

#### GET /health
Überprüft den System-Status.

**Response:**
```json
{
  "status": "string",
  "components": {
    "model": "healthy",
    "database": "healthy",
    "embeddings": "healthy"
  },
  "metrics": {
    "uptime": float,
    "requests_processed": integer,
    "current_load": float
  }
}
```

## Fehler-Responses

Alle Fehler folgen diesem Format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": object (optional)
  }
}
```

### Häufige Fehlercodes

- `400`: Ungültige Anfrage
- `401`: Nicht authentifiziert
- `403`: Nicht autorisiert
- `404`: Ressource nicht gefunden
- `429`: Zu viele Anfragen
- `500`: Interner Serverfehler

## Rate Limiting

- 100 Anfragen pro Minute pro IP
- 1000 Anfragen pro Stunde pro API-Key

## Beispiele

### Dokument hinzufügen

```python
import requests

url = "http://your-domain.com/api/v1/documents"
headers = {
    "Authorization": "Bearer your_token",
    "Content-Type": "application/json"
}
data = {
    "content": "Beispieltext für das Dokument",
    "metadata": {
        "source": "internal-kb",
        "type": "documentation",
        "tags": ["python", "api"]
    }
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Abfrage ausführen

```python
import requests

url = "http://your-domain.com/api/v1/query"
headers = {
    "Authorization": "Bearer your_token",
    "Content-Type": "application/json"
}
data = {
    "text": "Wie installiere ich das System?",
    "context_size": 3
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## Best Practices

1. Chunking
- Optimale Chunk-Größe: 512-1024 Tokens
- Überlappung: 10-20% der Chunk-Größe

2. Embeddings
- Cache häufig verwendete Embeddings
- Batch-Verarbeitung für große Dokumentenmengen

3. Abfragen
- Nutzen Sie spezifische Queries
- Fügen Sie relevanten Kontext hinzu

4. Performance
- Implementieren Sie Caching
- Nutzen Sie Batch-Verarbeitung
- Monitoring der Token-Nutzung

## Änderungsprotokoll

### v1.0.0 (2025-02-13)
- Initiale API-Version
- RAG-System Integration
- ChromaDB Backend
- DeepSeek-R1 Integration