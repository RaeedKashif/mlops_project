import torch
import torch.nn as nn
import torch.nn.functional as F

class HealthRiskPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=2):
        super(HealthRiskPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class WeatherPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1):
        super(WeatherPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_model_params(model):
    """Get model parameters as a list of numpy arrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_params(model, params):
    """Set model parameters from a list of numpy arrays"""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)