#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Starting Compliance Copilot Docker Container${NC}"
echo "================================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for services (if needed in future)
wait_for_service() {
    local service_name=$1
    local service_host=$2
    local service_port=$3
    local timeout=${4:-30}

    echo -e "${YELLOW}⏳ Waiting for $service_name to be available...${NC}"

    for i in $(seq 1 $timeout); do
        if timeout 1 bash -c "</dev/tcp/$service_host/$service_port" 2>/dev/null; then
            echo -e "${GREEN}✅ $service_name is available!${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done

    echo -e "${RED}❌ $service_name is not available after ${timeout}s${NC}"
    return 1
}

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p /app/data /app/logs /app/config /app/templates /app/static

# Set proper permissions
chown -R appuser:appuser /app/data /app/logs 2>/dev/null || true

# Check if .env file exists
if [ ! -f "/app/.env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Creating minimal configuration...${NC}"
    cat > /app/.env << EOF
# Minimal Docker configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./data/auditpilot.db

# Add your API keys here
GEMINI_API_KEY=your-gemini-api-key-here
PORTIA_API_KEY=your-portia-api-key-here

GITHUB_TOKEN=
GITHUB_WEBHOOK_SECRET=
EOF
    echo -e "${YELLOW}📝 Please update /app/.env with your actual API keys${NC}"
fi

# Check Python version
echo -e "${BLUE}🐍 Python version: $(python --version)${NC}"

# Run any pre-startup checks
echo -e "${BLUE}🔍 Running pre-startup checks...${NC}"

# Check if essential environment variables are set
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your-gemini-api-key-here" ]; then
    echo -e "${YELLOW}⚠️  GEMINI_API_KEY not set or using default value${NC}"
fi

if [ -z "$PORTIA_API_KEY" ] || [ "$PORTIA_API_KEY" = "your-portia-api-key-here" ]; then
    echo -e "${YELLOW}⚠️  PORTIA_API_KEY not set or using default value${NC}"
fi

# Initialize database if needed
if [ ! -f "/app/data/auditpilot.db" ]; then
    echo -e "${BLUE}🗄️  Initializing database...${NC}"
    python -c "
try:
    from src.models import create_database
    from src.config import settings
    create_database(settings.database_url)
    print('✅ Database initialized successfully')
except Exception as e:
    print(f'⚠️  Database initialization warning: {e}')
" || echo -e "${YELLOW}⚠️  Database initialization had issues, but continuing...${NC}"
fi

# If first argument is run.py or starts with python, run the application
if [ "$1" = "run.py" ] || [[ "$1" == python* ]] || [ $# -eq 0 ]; then
    echo -e "${GREEN}🚀 Starting Compliance Copilot application...${NC}"
    echo "================================================"
    echo -e "${GREEN}📊 Dashboard will be available at: http://localhost:8000${NC}"
    echo -e "${GREEN}📚 API Documentation: http://localhost:8000/docs${NC}"
    echo "================================================"

    # Start the application
    exec python run.py
else
    # Run custom command
    echo -e "${BLUE}🔧 Running custom command: $*${NC}"
    exec "$@"
fi
