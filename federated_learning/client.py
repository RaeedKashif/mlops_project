import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import HealthRiskPredictor, WeatherPredictor, get_model_params, set_model_params
from utils import get_client_data, HealthDataset, train_model, test_model
import sys
import os

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, db_path: str):
        self.client_id = client_id
        self.db_path = db_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model type based on client ID
        if "hospital" in client_id:
            self.model_type = "health"
            self.model = HealthRiskPredictor().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.model_type = "weather"
            self.model = WeatherPredictor().to(self.device)
            self.criterion = nn.MSELoss()  # Regression for weather prediction
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"Client {client_id} initialized with {self.model_type} model")
    
    def get_parameters(self, config):
        return get_model_params(self.model)
    
    def set_parameters(self, parameters):
        set_model_params(self.model, parameters)
    
    def fit(self, parameters, config):
        """Train the model on the client's data"""
        print(f"Client {self.client_id} starting training...")
        
        # Set model parameters from server
        self.set_parameters(parameters)
        
        # Get client data
        features, labels = get_client_data(self.db_path, self.client_id, self.model_type)
        
        if features is None or len(features) < 10:
            print(f"Client {self.client_id}: Not enough data for training")
            return self.get_parameters({}), 0, {}
        
        # Create dataset and dataloader
        dataset = HealthDataset(features, labels)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        self.model = train_model(
            self.model, train_loader, self.criterion, self.optimizer, self.device, epochs=2
        )
        
        # Return updated parameters
        print(f"Client {self.client_id} training completed")
        return self.get_parameters({}), len(dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the client's data"""
        self.set_parameters(parameters)
        
        features, labels = get_client_data(self.db_path, self.client_id, self.model_type)
        
        if features is None:
            return 0.0, 0, {"accuracy": 0.0}
        
        dataset = HealthDataset(features, labels)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        loss, accuracy = test_model(self.model, test_loader, self.criterion, self.device)
        
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return loss, len(dataset), {"accuracy": accuracy}

def main():
    client_id = os.getenv("CLIENT_ID", "default_client")
    db_path = os.getenv("DB_PATH", "/app/db/events.db")
    server_address = os.getenv("SERVER_ADDRESS", "localhost:8080")
    
    # Start Flower client
    client = FlowerClient(client_id, db_path)
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    main()