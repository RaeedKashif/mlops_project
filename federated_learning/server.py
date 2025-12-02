import flwr as fl
import torch
from model import HealthRiskPredictor, WeatherPredictor, get_model_params, set_model_params
from utils import get_available_clients
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import os

class FederatedServer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.health_model = HealthRiskPredictor()
        self.weather_model = WeatherPredictor()
        
        # Track training history
        self.training_history = {
            'health': {'loss': [], 'accuracy': []},
            'weather': {'loss': [], 'accuracy': []}
        }
        
    def get_initial_parameters(self, model_type: str = "health"):
        """Get initial model parameters"""
        if model_type == "health":
            return get_model_params(self.health_model)
        else:
            return get_model_params(self.weather_model)
    
    def aggregate_health_metrics(self, results: List[Tuple]) -> Tuple[float, Dict]:
        """Aggregate health model metrics"""
        losses = [r[0] for r in results if r[0] is not None]
        accuracies = [r[1] for r in results if r[1] is not None]
        num_examples = [r[2] for r in results]
        
        # Federated Averaging
        total_examples = sum(num_examples)
        if total_examples == 0:
            return 0.0, {}
            
        # Weighted average based on dataset size
        weights = [n / total_examples for n in num_examples]
        aggregated_loss = sum(l * w for l, w in zip(losses, weights))
        aggregated_accuracy = sum(a * w for a, w in zip(accuracies, weights))
        
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
    server = FederatedServer(db_path)
    
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