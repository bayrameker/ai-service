#!/bin/bash

# AI Service Setup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 AI Service Setup Script${NC}"
echo -e "${BLUE}This script will help you set up the AI Service environment${NC}"

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

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running on Windows (Git Bash/WSL)
check_platform() {
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        PLATFORM="windows"
        print_info "Detected Windows platform"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PLATFORM="linux"
        print_info "Detected Linux platform"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        PLATFORM="macos"
        print_info "Detected macOS platform"
    else
        PLATFORM="unknown"
        print_warning "Unknown platform: $OSTYPE"
    fi
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python 3 is installed (version: $PYTHON_VERSION)"
        
        # Check if version is 3.9+
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_status "Python version is compatible (3.9+)"
        else
            print_error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        print_info "Please install Python 3.9+ from https://python.org"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_status "pip3 is available"
    else
        print_error "pip3 is not installed"
        exit 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        print_status "Git is available"
    else
        print_error "Git is not installed"
        print_info "Please install Git from https://git-scm.com"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_status "Docker is available"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker is not installed (optional for development)"
        print_info "Install Docker from https://docker.com for containerized deployment"
        DOCKER_AVAILABLE=false
    fi
    
    # Check Docker Compose (optional)
    if command -v docker-compose &> /dev/null; then
        print_status "Docker Compose is available"
        DOCKER_COMPOSE_AVAILABLE=true
    else
        print_warning "Docker Compose is not installed (optional for development)"
        DOCKER_COMPOSE_AVAILABLE=false
    fi
}

# Setup Python virtual environment
setup_virtual_environment() {
    echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ "$PLATFORM" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "pip upgraded"
    
    cd ..
}

# Install Python dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    cd backend
    
    # Activate virtual environment
    if [[ "$PLATFORM" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Install requirements
    pip install -r requirements.txt
    print_status "Python dependencies installed"
    
    cd ..
}

# Setup environment configuration
setup_environment() {
    echo -e "${YELLOW}Setting up environment configuration...${NC}"
    
    # Create .env file if it doesn't exist
    if [ ! -f "backend/.env" ]; then
        cat > backend/.env << EOF
# Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Database Configuration
REDIS_URL=redis://localhost:6379/0
VECTOR_DB_TYPE=faiss
VECTOR_DB_PATH=./data/vector_db

# Security
JWT_SECRET_KEY=your-secret-key-change-in-production

# LLM Provider API Keys (add your keys here)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Optional: Database connections
DATABASE_URL=postgresql://user:password@localhost:5432/ai_service
MONGODB_URL=mongodb://localhost:27017/ai_service
EOF
        print_status "Environment file created at backend/.env"
        print_warning "Please add your API keys to backend/.env"
    else
        print_info "Environment file already exists"
    fi
}

# Setup data directories
setup_directories() {
    echo -e "${YELLOW}Setting up data directories...${NC}"
    
    # Create necessary directories
    mkdir -p backend/data/vector_db
    mkdir -p backend/logs
    mkdir -p backend/plugins/external
    mkdir -p backend/plugins/builtin
    
    print_status "Data directories created"
}

# Setup Redis (if Docker is available)
setup_redis() {
    if [ "$DOCKER_AVAILABLE" = true ]; then
        echo -e "${YELLOW}Setting up Redis with Docker...${NC}"
        
        # Check if Redis container is already running
        if docker ps | grep -q redis; then
            print_info "Redis container is already running"
        else
            # Start Redis container
            docker run -d --name ai-service-redis -p 6379:6379 redis:7-alpine
            print_status "Redis container started"
        fi
    else
        print_warning "Docker not available - please install Redis manually"
        print_info "Install Redis from https://redis.io/download"
    fi
}

# Run initial tests
run_tests() {
    echo -e "${YELLOW}Running initial tests...${NC}"
    
    cd backend
    
    # Activate virtual environment
    if [[ "$PLATFORM" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Run basic tests
    python -m pytest tests/ -v --tb=short -x
    
    if [ $? -eq 0 ]; then
        print_status "Initial tests passed"
    else
        print_warning "Some tests failed - this is normal for initial setup"
        print_info "You may need to configure API keys and services"
    fi
    
    cd ..
}

# Create sample plugin
create_sample_plugin() {
    echo -e "${YELLOW}Creating sample plugin...${NC}"
    
    mkdir -p backend/plugins/builtin/sample_plugin
    
    cat > backend/plugins/builtin/sample_plugin/plugin.json << EOF
{
    "name": "Sample Plugin",
    "version": "1.0.0",
    "description": "A sample plugin for demonstration",
    "author": "AI Service Team",
    "plugin_type": "tool",
    "dependencies": [],
    "min_system_version": "1.0.0",
    "entry_point": "plugins.builtin.sample_plugin.main",
    "hooks": ["before_request"],
    "enabled": true
}
EOF

    cat > backend/plugins/builtin/sample_plugin/main.py << EOF
"""
Sample Plugin for AI Service
"""

from app.plugins.manager import BasePlugin, PluginMetadata


class SamplePlugin(BasePlugin):
    """Sample plugin implementation"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Sample Plugin",
            version="1.0.0",
            description="A sample plugin for demonstration",
            author="AI Service Team",
            plugin_type="tool",
            entry_point="plugins.builtin.sample_plugin.main"
        )
    
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        self.logger.info("Sample plugin initialized")
        return True
    
    async def on_before_request(self, request_data):
        """Handle before request hook"""
        self.logger.info(f"Processing request: {request_data}")
        return request_data
    
    async def health_check(self) -> bool:
        """Check plugin health"""
        return True
EOF

    print_status "Sample plugin created"
}

# Show setup completion info
show_completion_info() {
    echo ""
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Add your API keys to backend/.env:"
    echo "   - OPENAI_API_KEY=your_openai_key"
    echo "   - ANTHROPIC_API_KEY=your_anthropic_key"
    echo ""
    echo "2. Start the development server:"
    echo "   cd backend"
    if [[ "$PLATFORM" == "windows" ]]; then
        echo "   source venv/Scripts/activate"
    else
        echo "   source venv/bin/activate"
    fi
    echo "   python -m uvicorn app.main:app --reload"
    echo ""
    echo "3. Access the API documentation:"
    echo "   http://localhost:8000/docs"
    echo ""
    echo -e "${YELLOW}Alternative: Use Docker Compose${NC}"
    if [ "$DOCKER_COMPOSE_AVAILABLE" = true ]; then
        echo "   docker-compose up -d"
        echo "   # Access at http://localhost:8000"
    else
        echo "   (Docker Compose not available)"
    fi
    echo ""
    echo -e "${YELLOW}Useful Commands:${NC}"
    echo "   ./scripts/deploy.sh dev     - Deploy in development mode"
    echo "   ./scripts/deploy.sh prod    - Deploy in production mode"
    echo "   python backend/scripts/run_tests.py - Run tests"
    echo ""
    echo -e "${BLUE}For more information, see README.md${NC}"
}

# Main setup flow
main() {
    check_platform
    check_prerequisites
    setup_virtual_environment
    install_dependencies
    setup_environment
    setup_directories
    setup_redis
    create_sample_plugin
    
    # Run tests (optional)
    read -p "Run initial tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    show_completion_info
}

# Handle script arguments
case "$1" in
    "clean")
        echo -e "${YELLOW}Cleaning up setup...${NC}"
        rm -rf backend/venv
        rm -f backend/.env
        rm -rf backend/data
        rm -rf backend/logs
        if [ "$DOCKER_AVAILABLE" = true ]; then
            docker stop ai-service-redis 2>/dev/null || true
            docker rm ai-service-redis 2>/dev/null || true
        fi
        print_status "Cleanup completed"
        exit 0
        ;;
    "help"|"-h"|"--help")
        echo "AI Service Setup Script"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  (no args)  - Run full setup"
        echo "  clean      - Clean up setup files"
        echo "  help       - Show this help message"
        exit 0
        ;;
    "")
        # Run main setup
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
