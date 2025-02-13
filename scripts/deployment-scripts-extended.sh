#!/bin/bash

# deployment.sh - Hauptdeployment-Skript mit erweiterten Funktionen
set -e

# Konfiguration
NAMESPACE="rag-system"
DEPLOYMENT_ENV=$1
VERSION=$(git describe --tags --always)
REGISTRY="your-registry.azurecr.io"

# Farben für Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
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

# Deployment Status prüfen
check_deployment_status() {
    local deployment=$1
    local timeout=$2
    local ready=false
    local counter=0
    
    while [ $counter -lt $timeout ]; do
        if kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=1s >/dev/null 2>&1; then
            ready=true
            break
        fi
        counter=$((counter + 1))
        sleep 1
    done
    
    if [ "$ready" = false ]; then
        error "Deployment $deployment nicht bereit nach $timeout Sekunden"
    fi
}

# GPU-Verfügbarkeit prüfen
check_gpu_availability() {
    if ! kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu | grep -q "[1-9]"; then
        error "Keine GPUs im Cluster verfügbar"
    fi
}

# Helm Charts updaten
update_helm_charts() {
    log "Aktualisiere Helm Charts..."
    
    # Prometheus Operator
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set grafana.enabled=true \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
        
    # ElasticSearch
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace $NAMESPACE \
        --set minimumMasterNodes=1
        
    # Kibana
    helm upgrade --install kibana elastic/kibana \
        --namespace $NAMESPACE \
        --set elasticsearch.hosts[0]=elasticsearch-master
}

# Backup erstellen
create_backup() {
    log "Erstelle Backup..."
    
    # ChromaDB Backup
    kubectl exec -n $NAMESPACE deployment/chroma -- \
        tar czf /backup/chroma-$(date +%Y%m%d).tar.gz /chroma/data
        
    # Konfiguration Backup
    kubectl get configmap -n $NAMESPACE -o yaml > backup/configmaps-$(date +%Y%m%d).yaml
    kubectl get secret -n $NAMESPACE -o yaml > backup/secrets-$(date +%Y%m%d).yaml
}

# Rollback durchführen
perform_rollback() {
    log "Führe Rollback durch..."
    
    # Vorheriges Image identifizieren
    local previous_version=$(kubectl get deployment model-deployment -n $NAMESPACE -o jsonpath='{.metadata.annotations.previous-version}')
    
    if [ -z "$previous_version" ]; then
        error "Keine vorherige Version gefunden"
    fi
    
    # Rollback der Deployments
    kubectl rollout undo deployment/model-deployment -n $NAMESPACE
    kubectl rollout undo deployment/api-deployment -n $NAMESPACE
    kubectl rollout undo deployment/ui-deployment -n $NAMESPACE
    
    # Status prüfen
    check_deployment_status "model-deployment" 300
    check_deployment_status "api-deployment" 180
    check_deployment_status "ui-deployment" 180
}

# Hauptdeployment-Prozess
main() {
    log "Starte Deployment für Umgebung: $DEPLOYMENT_ENV"
    
    # Umgebungsspezifische Konfiguration laden
    source "config/$DEPLOYMENT_ENV.env"
    
    # Voraussetzungen prüfen
    check_prerequisites
    check_gpu_availability
    
    # Backup erstellen
    create_backup
    
    # Helm Charts aktualisieren
    update_helm_charts
    
    # Deployments anwenden
    kubectl apply -k "overlays/$DEPLOYMENT_ENV"
    
    # Deployment-Status überprüfen
    check_deployment_status "model-deployment" 300
    check_deployment_status "api-deployment" 180
    check_deployment_status "ui-deployment" 180
    
    # Post-Deployment Tests
    if ! ./scripts/test_deployment.sh; then
        warn "Post-Deployment Tests fehlgeschlagen"
        if [ "$DEPLOYMENT_ENV" = "production" ]; then
            perform_rollback
            error "Deployment fehlgeschlagen - Rollback durchgeführt"
        fi
    fi
    
    log "Deployment erfolgreich abgeschlossen!"
}

# Skript ausführen
if [ -z "$DEPLOYMENT_ENV" ]; then
    error "Deployment-Umgebung nicht angegeben"
fi

main