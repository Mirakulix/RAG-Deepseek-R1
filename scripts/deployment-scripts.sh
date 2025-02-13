#!/bin/bash

# deploy.sh - Hauptdeployment-Skript
set -e

# Konfigurationsvariablen
NAMESPACE="rag-system"
REGISTRY="your-registry.azurecr.io"
VERSION=$(git describe --tags --always)

# Farben für Ausgabe
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Hilfsfunktionen
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Prüfe Voraussetzungen
check_prerequisites() {
    log "Prüfe Voraussetzungen..."
    
    # Kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl nicht gefunden"
    fi
    
    # Docker
    if ! command -v docker &> /dev/null; then
        error "docker nicht gefunden"
    fi
    
    # Helm
    if ! command -v helm &> /dev/null; then
        error "helm nicht gefunden"
    }
}

# Namespace erstellen/prüfen
setup_namespace() {
    log "Konfiguriere Namespace..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warn "Namespace $NAMESPACE existiert bereits"
    else
        kubectl create namespace "$NAMESPACE"
        log "Namespace $NAMESPACE erstellt"
    fi
}

# Docker Images bauen
build_images() {
    log "Baue Docker Images..."
    
    # Model Server
    docker build -t "$REGISTRY/model:$VERSION" ./src/model
    docker push "$REGISTRY/model:$VERSION"
    
    # API Server
    docker build -t "$REGISTRY/api:$VERSION" ./src/api
    docker push "$REGISTRY/api:$VERSION"
    
    # UI
    docker build -t "$REGISTRY/ui:$VERSION" ./src/ui
    docker push "$REGISTRY/ui:$VERSION"
}

# Secrets erstellen
create_secrets() {
    log "Erstelle Kubernetes Secrets..."
    
    # Prüfe ob .env existiert
    if [[ ! -f .env ]]; then
        error ".env Datei nicht gefunden"
    fi
    
    # Erstelle Secrets aus .env
    source .env
    kubectl create secret generic app-secrets \
        --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
        --from-literal=MODEL_TEMPERATURE="$MODEL_TEMPERATURE" \
        --from-literal=CHROMA_API_KEY="$CHROMA_API_KEY" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
}

# ChromaDB einrichten
setup_chroma() {
    log "Konfiguriere ChromaDB..."
    
    helm repo add chroma https://chroma-core.github.io/helm-charts
    helm repo update
    
    helm upgrade --install chroma chroma/chroma \
        --namespace "$NAMESPACE" \
        --set persistence.enabled=true \
        --set persistence.size=10Gi
}

# Deployments anwenden
apply_deployments() {
    log "Wende Deployments an..."
    
    # Ersetze Versionen in Deployment Files
    for file in kubernetes/deployments/*.yml; do
        sed -i "s|IMAGE_VERSION|$VERSION|g" "$file"
    done
    
    # Wende Deployments an
    kubectl apply -f kubernetes/deployments/ -n "$NAMESPACE"
    kubectl apply -f kubernetes/services/ -n "$NAMESPACE"
}

# Deployment überprüfen
verify_deployment() {
    log "Überprüfe Deployment..."
    
    # Warte auf Pods
    kubectl wait --for=condition=ready pod -l app=deepseek-model -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=fastapi-service -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=streamlit-ui -n "$NAMESPACE" --timeout=300s
    
    # Prüfe Services
    for service in deepseek-model-service fastapi-service streamlit-ui-service; do
        if ! kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            error "Service $service nicht gefunden"
        fi
    done
    
    log "Deployment erfolgreich!"
}

# Cleanup Funktion
cleanup() {
    warn "Cleanup nach Fehler..."
    # Hier können Cleanup-Aktionen definiert werden
}

# Hauptprozess
main() {
    trap cleanup ERR
    
    log "Starte Deployment..."
    
    check_prerequisites
    setup_namespace
    build_images
    create_secrets
    setup_chroma
    apply_deployments
    verify_deployment
    
    log "Deployment abgeschlossen!"
}

# Skript ausführen
main