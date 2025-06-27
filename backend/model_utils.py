import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

class LSTMSquatClassifier(nn.Module):
    def __init__(self, input_size=133, hidden_size=64, num_classes=7):
        super(LSTMSquatClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# Label mapping from the training
LABEL_MAP = {
    0: "good",
    1: "bothknees", 
    2: "buttwink",
    3: "halfsquat",
    4: "leanforward",
    5: "leftknee",
    6: "rightknee"
}

class SquatFormPredictor:
    def __init__(self, model_path="model/lstm_squat_model.pt", target_len=74):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_len = target_len
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained LSTM model"""
        try:
            # Create model instance
            self.model = LSTMSquatClassifier(input_size=133, hidden_size=64, num_classes=7)
            
            # Load the state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_csv(self, csv_path):
        """Preprocess CSV data for model prediction"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Keep only numeric columns (INCLUDE frame column to match training)
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Convert to numpy array
            data = numeric_df.to_numpy()
            
            # Handle sequence length normalization
            if data.shape[0] > self.target_len:
                # Truncate if too long
                data = data[:self.target_len]
            elif data.shape[0] < self.target_len:
                # Pad if too short
                pad_len = self.target_len - data.shape[0]
                pad = np.zeros((pad_len, data.shape[1]))
                data = np.vstack((data, pad))
            
            # Convert to tensor and add batch dimension
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            return tensor.to(self.device)
            
        except Exception as e:
            print(f"❌ Error preprocessing CSV: {e}")
            raise
    
    def predict(self, csv_path):
        """Make prediction on CSV data"""
        try:
            # Preprocess the data
            input_tensor = self.preprocess_csv(csv_path)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get label name
            predicted_label = LABEL_MAP[predicted_class]
            
            return {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "all_probabilities": probabilities[0].cpu().numpy().tolist(),
                "class_names": list(LABEL_MAP.values())
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise

def load_squat_model(model_path="model/lstm_squat_model.pt"):
    """Convenience function to load the model"""
    return SquatFormPredictor(model_path)
