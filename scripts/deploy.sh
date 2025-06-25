#!/bin/bash

# AI Service Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE="ai-service"
DOCKER_TAG="latest"
NAMESPACE="ai-service"
ENVIRONMENT=${1:-development}

echo -e "${GREEN}🚀 Starting AI Service Deployment${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_status "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_status "Docker Compose is available"
    
    # Check if running in Kubernetes environment
    if command -v kubectl &> /dev/null; then
        print_status "kubectl is available"
        KUBERNETES_AVAILABLE=true
    else
        print_warning "kubectl not found - Kubernetes deployment will be skipped"
        KUBERNETES_AVAILABLE=false
    fi
}

# Build Docker image
build_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker build --target production -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    else
        docker build --target development -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
    fi
    
    print_status "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    echo -e "${YELLOW}Deploying with Docker Compose...${NC}"
    
    # Stop existing containers
    docker-compose down
    
    # Start services
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    print_status "Services started with Docker Compose"
    
    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 30
    
    # Check health
    check_health_docker_compose
}

# Deploy to Kubernetes
deploy_kubernetes() {
    if [ "$KUBERNETES_AVAILABLE" = false ]; then
        print_warning "Skipping Kubernetes deployment - kubectl not available"
        return
    fi
    
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Apply configurations
    kubectl apply -f k8s/configmap.yaml
    
    # Deploy databases
    kubectl apply -f k8s/redis.yaml
    kubectl apply -f k8s/postgres.yaml
    
    # Wait for databases to be ready
    echo -e "${YELLOW}Waiting for databases to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n ${NAMESPACE}
    
    # Deploy AI service
    kubectl apply -f k8s/ai-service.yaml
    
    # Wait for deployment
    kubectl wait --for=condition=available --timeout=300s deployment/ai-service -n ${NAMESPACE}
    
    print_status "Kubernetes deployment completed"
    
    # Check health
    check_health_kubernetes
}

# Check health for Docker Compose
check_health_docker_compose() {
    echo -e "${YELLOW}Checking service health...${NC}"
    
    # Check AI service health
    for i in {1..10}; do
        if curl -f http://localhost:8000/api/v1/health/ > /dev/null 2>&1; then
            print_status "AI Service is healthy"
            break
        else
            if [ $i -eq 10 ]; then
                print_error "AI Service health check failed"
                docker-compose logs ai-service
                exit 1
            fi
            echo "Waiting for AI Service... (attempt $i/10)"
            sleep 10
        fi
    done
    
    # Check Redis
    if docker-compose exec redis redis-cli ping | grep -q PONG; then
        print_status "Redis is healthy"
    else
        print_error "Redis health check failed"
    fi
    
    # Check PostgreSQL
    if docker-compose exec postgres pg_isready -U ai_service > /dev/null 2>&1; then
        print_status "PostgreSQL is healthy"
    else
        print_error "PostgreSQL health check failed"
    fi
}

# Check health for Kubernetes
check_health_kubernetes() {
    echo -e "${YELLOW}Checking Kubernetes service health...${NC}"
    
    # Get service URL
    if kubectl get ingress ai-service-ingress -n ${NAMESPACE} > /dev/null 2>&1; then
        SERVICE_URL=$(kubectl get ingress ai-service-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}')
        SERVICE_URL="https://${SERVICE_URL}"
    else
        # Use port-forward for testing
        kubectl port-forward service/ai-service-service 8080:8000 -n ${NAMESPACE} &
        PORT_FORWARD_PID=$!
        SERVICE_URL="http://localhost:8080"
        sleep 5
    fi
    
    # Check health
    for i in {1..10}; do
        if curl -f ${SERVICE_URL}/api/v1/health/ > /dev/null 2>&1; then
            print_status "AI Service is healthy at ${SERVICE_URL}"
            break
        else
            if [ $i -eq 10 ]; then
                print_error "AI Service health check failed"
                kubectl logs -l app=ai-service -n ${NAMESPACE} --tail=50
                exit 1
            fi
            echo "Waiting for AI Service... (attempt $i/10)"
            sleep 10
        fi
    done
    
    # Clean up port-forward if used
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Run tests
run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    
    if [ "$ENVIRONMENT" = "development" ]; then
        # Run tests in development environment
        docker-compose exec ai-service python -m pytest tests/ -v
        print_status "Tests completed"
    else
        print_warning "Skipping tests in production environment"
    fi
}

# Show deployment info
show_deployment_info() {
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Service Information:${NC}"
    
    if [ "$KUBERNETES_AVAILABLE" = true ]; then
        echo "Kubernetes Deployment:"
        echo "  Namespace: ${NAMESPACE}"
        echo "  Pods: $(kubectl get pods -n ${NAMESPACE} -l app=ai-service --no-headers | wc -l)"
        echo "  Service: $(kubectl get service ai-service-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}'):8000"
        
        if kubectl get ingress ai-service-ingress -n ${NAMESPACE} > /dev/null 2>&1; then
            INGRESS_HOST=$(kubectl get ingress ai-service-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}')
            echo "  External URL: https://${INGRESS_HOST}"
        fi
    else
        echo "Docker Compose Deployment:"
        echo "  AI Service: http://localhost:8000"
        echo "  API Documentation: http://localhost:8000/docs"
        echo "  Redis: localhost:6379"
        echo "  PostgreSQL: localhost:5432"
        echo "  Grafana: http://localhost:3000 (admin/admin)"
        echo "  Prometheus: http://localhost:9090"
    fi
    
    echo ""
    echo -e "${YELLOW}Useful Commands:${NC}"
    if [ "$KUBERNETES_AVAILABLE" = true ]; then
        echo "  View logs: kubectl logs -l app=ai-service -n ${NAMESPACE} -f"
        echo "  Scale service: kubectl scale deployment ai-service --replicas=5 -n ${NAMESPACE}"
        echo "  Port forward: kubectl port-forward service/ai-service-service 8000:8000 -n ${NAMESPACE}"
    else
        echo "  View logs: docker-compose logs -f ai-service"
        echo "  Scale service: docker-compose up -d --scale ai-service=3"
        echo "  Stop services: docker-compose down"
    fi
}

# Main deployment flow
main() {
    check_prerequisites
    build_image
    
    if [ "$ENVIRONMENT" = "kubernetes" ] || [ "$KUBERNETES_AVAILABLE" = true ]; then
        deploy_kubernetes
    else
        deploy_docker_compose
    fi
    
    if [ "$ENVIRONMENT" = "development" ]; then
        run_tests
    fi
    
    show_deployment_info
}

# Handle script arguments
case "$1" in
    "development"|"dev")
        ENVIRONMENT="development"
        ;;
    "production"|"prod")
        ENVIRONMENT="production"
        ;;
    "kubernetes"|"k8s")
        ENVIRONMENT="kubernetes"
        ;;
    "test")
        echo -e "${YELLOW}Running tests only...${NC}"
        docker-compose exec ai-service python -m pytest tests/ -v
        exit 0
        ;;
    "clean")
        echo -e "${YELLOW}Cleaning up...${NC}"
        docker-compose down -v
        docker system prune -f
        if [ "$KUBERNETES_AVAILABLE" = true ]; then
            kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        fi
        print_status "Cleanup completed"
        exit 0
        ;;
    "help"|"-h"|"--help")
        echo "AI Service Deployment Script"
        echo ""
        echo "Usage: $0 [ENVIRONMENT]"
        echo ""
        echo "Environments:"
        echo "  development, dev  - Deploy in development mode with Docker Compose"
        echo "  production, prod  - Deploy in production mode"
        echo "  kubernetes, k8s   - Deploy to Kubernetes cluster"
        echo ""
        echo "Commands:"
        echo "  test             - Run tests only"
        echo "  clean            - Clean up all deployments"
        echo "  help             - Show this help message"
        exit 0
        ;;
    "")
        # Default to development
        ENVIRONMENT="development"
        ;;
    *)
        print_error "Unknown environment: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

# Run main deployment
main
