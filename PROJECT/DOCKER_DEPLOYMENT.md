# 🐳 ClimaCast Docker Deployment Guide

## Prerequisites

- Docker Desktop installed
- Docker Compose installed (comes with Docker Desktop)

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build
```

### 2. Access the Application

- **Application**: http://localhost:8501
- **With Nginx**: http://localhost:80 (if using production profile)

### 3. Stop the Application

```bash
# Stop containers
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Individual Docker Commands

### Build Docker Image

```bash
docker build -t climacast .
```

### Run Container

```bash
docker run -p 8501:8501 climacast
```

### Run with Volume Mounts

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/exports:/app/exports \
  -v $(pwd)/logs:/app/logs \
  climacast
```

## Production Deployment

### With Nginx Reverse Proxy

```bash
# Start with production profile
docker-compose --profile production up -d --build
```

This will start:
- ClimaCast application on port 8501
- Nginx reverse proxy on port 80

### Environment Variables

Create a `.env` file for production:

```env
# Database
DATABASE_URL=sqlite:///app/climacast.db

# API Keys (optional)
OPENWEATHER_API_KEY=your_key_here
NASA_API_KEY=your_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/climacast.log
```

## Cloud Deployment

### AWS ECS

1. Push image to ECR
2. Create ECS task definition
3. Deploy service

### Google Cloud Run

```bash
# Build and push to GCR
docker build -t gcr.io/PROJECT_ID/climacast .
docker push gcr.io/PROJECT_ID/climacast

# Deploy to Cloud Run
gcloud run deploy climacast \
  --image gcr.io/PROJECT_ID/climacast \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry myregistry --image climacast .

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name climacast \
  --image myregistry.azurecr.io/climacast \
  --ports 8501 \
  --dns-name-label climacast-app
```

## Monitoring and Logs

### View Logs

```bash
# View application logs
docker-compose logs -f climacast

# View all logs
docker-compose logs -f
```

### Health Check

The application includes health checks:
- **Endpoint**: http://localhost:8501/_stcore/health
- **Interval**: 30 seconds
- **Timeout**: 10 seconds

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8502:8501"  # Use different host port
   ```

2. **Permission issues**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

3. **Database not found**
   ```bash
   # Ensure database exists
   ls -la climacast.db
   ```

### Debug Mode

```bash
# Run with debug output
docker-compose up --build --force-recreate
```

## Scaling

### Horizontal Scaling

```bash
# Scale to multiple instances
docker-compose up --scale climacast=3
```

### Load Balancing

Use the nginx configuration for load balancing multiple instances.

## Security Considerations

1. **Environment Variables**: Never commit API keys
2. **Network Security**: Use proper firewall rules
3. **Image Security**: Regularly update base images
4. **Secrets Management**: Use Docker secrets or external secret managers

## Performance Optimization

1. **Multi-stage builds**: Reduce image size
2. **Layer caching**: Optimize Dockerfile layer order
3. **Resource limits**: Set memory and CPU limits
4. **Health checks**: Monitor application health

---

## 🚀 Ready to Deploy!

Your ClimaCast application is now containerized and ready for deployment anywhere Docker is supported!
