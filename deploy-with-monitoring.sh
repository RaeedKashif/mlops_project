#!/bin/bash

echo "🚀 Deploying IoT System with Monitoring..."

# Create directory structure
mkdir -p monitoring/grafana/provisioning/{datasources,dashboards}
mkdir -p data
mkdir -p ingestion/db
mkdir -p federated_learning/models

# Download sample data if not exists
if [ ! -f "data/heart.csv" ]; then
    echo "📥 Downloading sample heart data..."
    wget -O data/heart.csv https://raw.githubusercontent.com/datasets/heart-disease/master/data/heart.csv
fi

if [ ! -f "data/weather.csv" ]; then
    echo "📥 Downloading sample weather data..."
    wget -O data/weather.csv https://raw.githubusercontent.com/datasets/weather-data/master/data/weather.csv
fi

# Create Prometheus config
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'ingestion-api'
    static_configs:
      - targets: ['ingestion_api:8001']

  - job_name: 'fl-server'
    static_configs:
      - targets: ['fl_server:8081']
EOF

# Create Grafana datasource
cat > monitoring/grafana/provisioning/datasources/datasource.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Start the system
echo "🚀 Starting all services..."
docker-compose up -d

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📊 Access your services:"
echo "   Dashboard:      http://localhost:8501"
echo "   Ingestion API:  http://localhost:8000"
echo "   Prometheus:     http://localhost:9090"
echo "   Grafana:        http://localhost:3000 (admin/admin123)"
echo "   cAdvisor:       http://localhost:8088"
echo ""
echo "📈 Monitoring endpoints:"
echo "   Metrics:        http://localhost:8001/metrics"
echo "   FL Metrics:     http://localhost:8081/metrics"
echo ""
echo "🛑 To stop: docker-compose down"