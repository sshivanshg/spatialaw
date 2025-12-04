import json

def restore_notebook():
    notebook_path = "Spatial_Awareness_Project.ipynb"
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    
    # 1. Restore Random Forest Data Loading
    for cell in cells:
        if "Patching Random Forest Data Loading" in "".join(cell.get("source", [])) or "X, feature_names = generate_synthetic_data()" in "".join(cell.get("source", [])):
            print("Restoring Random Forest Data Loading...")
            new_code = """
# Load Processed Data
try:
    X = np.load("data/processed/binary/features.npy")
    y = pd.read_csv("data/processed/binary/labels.csv")['label'].values
    with open("data/processed/binary/feature_names.json", 'r') as f:
        feature_names = json.load(f)
    print(f"Loaded Real Dataset: {X.shape[0]} samples.")
except FileNotFoundError:
    print("Error: Real data not found in data/processed/binary/")
    print("Please ensure you have processed the WiAR dataset.")
    raise

# Split Data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=np.arange(len(X))))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
"""
            cell["source"] = [line + "\n" for line in new_code.split("\n")]

    # 2. Restore CNN Data Loading
    for cell in cells:
        if "Robust Data Loading" in "".join(cell.get("source", [])):
            print("Restoring CNN Data Loading...")
            new_code = """
# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Real Windows
if Path("data/processed/windows").exists():
    full_dataset = CSIDataset("data/processed/windows")
    print(f"Loaded {len(full_dataset)} real samples.")
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Add Synthetic Empty Data (Optional, to balance if needed)
    # X_synth, y_synth = generate_synthetic_windows(n_samples=500)
    # synth_dataset = torch.utils.data.TensorDataset(X_synth, y_synth)
    # train_dataset = torch.utils.data.ConcatDataset([train_dataset, synth_dataset])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting Training...")
    epochs = 10
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"CNN Test Accuracy: {100 * correct / total:.2f}%")
    
    # Plot Loss
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.title('CNN Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

else:
    print("Error: Real data not found in data/processed/windows/")
    print("Skipping CNN experiment.")
"""
            cell["source"] = [line + "\n" for line in new_code.split("\n")]

    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("Notebook restored to use real data.")

if __name__ == "__main__":
    restore_notebook()
