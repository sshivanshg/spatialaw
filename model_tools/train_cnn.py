import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import json

# --- Configuration ---
DATA_DIR = Path("data/processed/windows")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "presence_detector_cnn.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# --- Dataset ---
class CSIDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

def load_data():
    print("Loading data...")
    labels_df = pd.read_csv(DATA_DIR / "labels.csv")
    
    windows = []
    labels = []
    groups = []
    
    for _, row in labels_df.iterrows():
        file_path = DATA_DIR / row['window_file']
        if file_path.exists():
            try:
                w = np.load(file_path)
                # Ensure shape is (Channels, Time)
                if w.shape[0] not in [30, 60]: # If (Time, Channels)
                    w = w.T
                
                # Check for 60 channels (fix if needed)
                if w.shape[0] == 30:
                     # Duplicate to 60 if needed, or just accept 30. 
                     # But our previous fix said 60. Let's stick to what the data is.
                     pass

                windows.append(w)
                labels.append(row['label'])
                groups.append(row.get('source_recording', 'unknown'))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    windows = np.stack(windows)
    
    # Binarize labels: 0 is Empty, >0 is Presence
    # Note: In our processed data, label 0 is "No Activity" (from low motion or synthetic)
    # All other labels are activities.
    binary_labels = np.array([0 if l == 0 else 1 for l in labels])
    
    print(f"Loaded {len(windows)} windows. Shape: {windows.shape}")
    print(f"Class balance: {np.bincount(binary_labels)}")
    
    return windows, binary_labels, np.array(groups)

# --- Model ---
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=60, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Training ---
def train():
    windows, labels, groups = load_data()
    
    # Detect input channels from data
    input_channels = windows.shape[1]
    print(f"Detected input channels: {input_channels}")
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(windows, labels, groups))
    
    X_train, X_test = windows[train_idx], windows[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    train_dataset = CSIDataset(X_train, y_train)
    test_dataset = CSIDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = SimpleCNN(input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Acc: {100*correct/total:.2f}%")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    print(f"Test Accuracy: {100*correct/total:.2f}%")
    
    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
