import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

class HealthDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_client_data(db_path, client_id, data_type="heart"):
    """Get data for a specific client/node"""
    conn = sqlite3.connect(db_path)
    
    if data_type == "heart":
        query = """
            SELECT payload, node_id 
            FROM events 
            WHERE type = 'heart' AND node_id = ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """
    else:  # weather
        query = """
            SELECT payload, node_id 
            FROM events 
            WHERE type = 'weather' AND node_id = ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """
    
    df = pd.read_sql_query(query, conn, params=(client_id,))
    conn.close()
    
    if df.empty:
        return None, None
    
    # Parse payload and extract features
    features = []
    labels = []
    
    for _, row in df.iterrows():
        payload = json.loads(row['payload'])
        
        if data_type == "heart":
            # Heart risk prediction features
            heart_rate = payload.get('heart_rate')
            cholesterol = payload.get('cholesterol')
            blood_pressure = payload.get('blood_pressure')
            
            if all(v is not None for v in [heart_rate, cholesterol, blood_pressure]):
                # Simple risk calculation based on medical thresholds
                risk = 0
                if heart_rate > 100 or heart_rate < 60:  # Abnormal heart rate
                    risk += 1
                if cholesterol > 200:  # High cholesterol
                    risk += 1
                if blood_pressure > 140:  # High blood pressure
                    risk += 1
                
                features.append([heart_rate, cholesterol, blood_pressure])
                labels.append(min(risk, 1))  # Binary classification: 0=low risk, 1=high risk
                
        else:  # weather
            temperature = payload.get('temperature')
            humidity = payload.get('humidity')
            
            if temperature is not None and humidity is not None:
                # Weather anomaly detection
                features.append([temperature, humidity])
                # Simple anomaly: extreme temperatures or humidity
                is_anomaly = 1 if (temperature > 90 or temperature < 32 or humidity > 90) else 0
                labels.append(is_anomaly)
    
    if not features:
        return None, None
        
    return np.array(features), np.array(labels)

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    """Train model for one round"""
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    return model

def test_model(model, test_loader, criterion, device):
    """Test model performance"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0
    
    return avg_loss, accuracy

def get_available_clients(db_path):
    """Get list of available clients with data"""
    conn = sqlite3.connect(db_path)
    
    # Get clients with heart data
    heart_clients = pd.read_sql_query("""
        SELECT DISTINCT node_id 
        FROM events 
        WHERE type = 'heart' 
        GROUP BY node_id 
        HAVING COUNT(*) > 10
    """, conn)
    
    # Get clients with weather data
    weather_clients = pd.read_sql_query("""
        SELECT DISTINCT node_id 
        FROM events 
        WHERE type = 'weather' 
        GROUP BY node_id 
        HAVING COUNT(*) > 10
    """, conn)
    
    conn.close()
    
    return {
        'heart': heart_clients['node_id'].tolist(),
        'weather': weather_clients['node_id'].tolist()
    }