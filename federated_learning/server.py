# federated_learning/server.py
import flwr as fl
import torch
from model import HealthRiskPredictor, WeatherPredictor, get_model_params, set_model_params
from utils import get_available_clients
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import os
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import threading

# Prometheus metrics for FL Server
FL_ROUNDS_TOTAL = Counter('fl_rounds_total', 'Total federated learning rounds')
FL_CLIENTS_CONNECTED = Gauge('fl_clients_connected', 'Number of connected clients')
FL_MODEL_ACCURACY = Gauge('fl_model_accuracy', 'Model accuracy')
FL_MODEL_LOSS = Gauge('fl_model_loss', 'Model loss')
FL_TRAINING_EXAMPLES = Counter('fl_training_examples_total', 'Total training examples processed')
FL_AGGREGATION_TIME = Histogram('fl_aggregation_time_seconds', 'Time taken to aggregate models')
FL_CLIENT_DURATION = Histogram('fl_client_training_seconds', 'Client training duration', ['client_id'])

class FederatedServer:
    def __init__(self, db_path: str, metrics_port: int = 8081):
        self.db_path = db_path
        self.health_model = HealthRiskPredictor()
        self.weather_model = WeatherPredictor()
        
        # Track training history
        self.training_history = {
            'health': {'loss': [], 'accuracy': []},
            'weather': {'loss': [], 'accuracy': []}
        }
        
        # Start metrics server in background
        if metrics_port:
            threading.Thread(target=start_http_server, args=(metrics_port,), daemon=True).start()
            print(f"✅ Prometheus metrics server started on port {metrics_port}")
    
    def get_initial_parameters(self, model_type: str = "health"):
        """Get initial model parameters"""
        if model_type == "health":
            return get_model_params(self.health_model)
        else:
            return get_model_params(self.weather_model)
    
    def aggregate_health_metrics(self, results: List[Tuple]) -> Tuple[float, Dict]:
        """Aggregate health model metrics"""
        with FL_AGGREGATION_TIME.time():
            losses = [r[0] for r in results if r[0] is not None]
            accuracies = [r[1] for r in results if r[1] is not None]
            num_examples = [r[2] for r in results]
            
            # Update metrics
            FL_CLIENTS_CONNECTED.set(len(results))
            total_examples = sum(num_examples)
            FL_TRAINING_EXAMPLES.inc(total_examples)
            
            if total_examples == 0:
                return 0.0, {}
                
            # Weighted average based on dataset size
            weights = [n / total_examples for n in num_examples]
            aggregated_loss = sum(l * w for l, w in zip(losses, weights))
            aggregated_accuracy = sum(a * w for a, w in zip(accuracies, weights))
            
            # Update model metrics
            FL_MODEL_LOSS.set(aggregated_loss)
            FL_MODEL_ACCURACY.set(aggregated_accuracy)
            
            metrics = {
                "aggregated_loss": aggregated_loss,
                "aggregated_accuracy": aggregated_accuracy,
                "total_examples": total_examples,
                "num_clients": len(results)
            }
            
            return aggregated_loss, metrics

class HealthStrategy(fl.server.strategy.FedAvg):
    def __init__(self, server: FederatedServer, **kwargs):
        super().__init__(**kwargs)
        self.server = server
    
    def aggregate_fit(self, rnd, results, failures):
        """Aggregate model parameters using federated averaging"""
        FL_ROUNDS_TOTAL.inc()
        
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            # Update server model with aggregated parameters
            set_model_params(self.server.health_model, aggregated_parameters[0])
            
            # Save model checkpoint
            os.makedirs('/app/models', exist_ok=True)
            torch.save(self.server.health_model.state_dict(), f'/app/models/health_model_round_{rnd}.pth')
            
            print(f"Round {rnd}: Model aggregated and saved")
        
        return aggregated_parameters

def main():
    # Initialize server
    db_path = os.getenv("DB_PATH", "/app/db/events.db")
    metrics_port = int(os.getenv("PROMETHEUS_PORT", "8081"))
    server = FederatedServer(db_path, metrics_port)
    
    # Get available clients
    available_clients = get_available_clients(db_path)
    print(f"Available clients: {available_clients}")
    
    # Configure strategy
    strategy = HealthStrategy(
        server=server,
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
    
    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

if __name__ == "__main__":
    main()