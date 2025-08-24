# Docker Guide for Compliance Copilot

This guide explains how to run Compliance Copilot using Docker for both development and production environments. For local development without Docker, see the [Local Development Setup](#local-development-setup) section.

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed on your system
- Your `.env` file configured with necessary API keys

### Production Deployment

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Run in detached mode:**
   ```bash
   docker-compose up -d --build
   ```

3. **Access the application:**
   - Dashboard: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Development Setup

For development with live code reloading:

1. **Start in development mode:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.override.yml up --build
   ```

   Or simply:
   ```bash
   docker-compose up --build  # override is loaded automatically
   ```

2. **The development setup includes:**
   - Live code reloading when Python files change
   - Volume mounts for real-time development
   - Debug logging enabled
   - Faster rebuilds using multi-stage builds

## Docker Files Overview

### Core Files
- `Dockerfile` - Multi-stage production-optimized container
- `docker-compose.yml` - Production configuration
- `docker-compose.override.yml` - Development overrides (auto-loaded)
- `docker-entrypoint.sh` - Initialization and startup script
- `.dockerignore` - Excludes unnecessary files from build context

### Requirements
- `requirements.txt` - Pinned production dependencies
- `requirements-prod.txt` - Backup with exact versions

## Configuration

### Environment Variables

The container uses the following environment variables (via `.env` file):

```env
# Application
DEBUG=false
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./data/auditpilot.db

# Required API Keys
GEMINI_API_KEY=your-gemini-api-key-here
PORTIA_API_KEY=your-portia-api-key-here

GITHUB_TOKEN=your-github-token
GITHUB_WEBHOOK_SECRET=your-webhook-secret
```

### Volume Mounts

The following directories are mounted for persistence:
- `./data:/app/data` - Database and persistent data
- `./logs:/app/logs` - Application logs
- `./config:/app/config` - Configuration files

## Docker Commands Reference

### Building
```bash
# Build the image
docker-compose build

# Build without cache
docker-compose build --no-cache

# Build specific service
docker-compose build compliance-copilot
```

### Running
```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# Start with specific compose file
docker-compose -f docker-compose.yml up

# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

### Management
```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Execute commands in running container
docker-compose exec compliance-copilot bash

# View running services
docker-compose ps
```

### Debugging
```bash
# Run shell in container
docker-compose run --rm compliance-copilot bash

# Run with custom command
docker-compose run --rm compliance-copilot python -c "from src.config import settings; print(settings.database_url)"

# Check container health
docker-compose exec compliance-copilot curl -f http://localhost:8000/health
```

## Production Deployment

### Docker Compose Production

For production, create a `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  compliance-copilot:
    image: compliance-copilot:latest
    restart: always
    environment:
      - DEBUG=false
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: compliance_copilot
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Container Registry

1. **Build for production:**
   ```bash
   docker build -t compliance-copilot:latest .
   ```

2. **Tag for registry:**
   ```bash
   docker tag compliance-copilot:latest your-registry/compliance-copilot:latest
   ```

3. **Push to registry:**
   ```bash
   docker push your-registry/compliance-copilot:latest
   ```

## Troubleshooting

### Common Issues

1. **Permission errors:**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data ./logs
   ```

2. **Port already in use:**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

3. **Database issues:**
   ```bash
   # Reset database
   docker-compose down -v
   docker-compose up --build
   ```

4. **API key issues:**
   ```bash
   # Check environment variables
   docker-compose exec compliance-copilot env | grep -E "(GEMINI|PORTIA)"
   ```

### Health Checks

The container includes health checks:
```bash
# Check container health
docker-compose ps
docker inspect compliance-copilot | grep Health -A 10
```

### Performance

- Multi-stage builds minimize image size
- Health checks ensure service availability
- Non-root user for security
- Optimized for production with pinned dependencies

## Development Tips

1. **Code changes are automatically reloaded** in development mode
2. **Use volume mounts** to persist data between container restarts
3. **Check logs regularly** with `docker-compose logs -f`
4. **Use the health endpoint** to verify the application is running correctly

## Security Notes

- Container runs as non-root user `appuser`
- Secrets should be provided via environment variables
- Database files are persisted in volumes
- Production images exclude development tools
