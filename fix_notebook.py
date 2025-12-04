import json

def fix_notebook():
    notebook_path = "Spatial_Awareness_Project.ipynb"
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    
    # 0. Fix Imports
    for cell in cells:
        if "import numpy as np" in "".join(cell.get("source", [])):
            print("Patching Imports...")
            new_source = cell["source"]
            # Add missing imports
            new_source.insert(0, "from dataclasses import dataclass\n")
            new_source.insert(0, "from typing import List, Sequence, Callable, Iterable\n")
            cell["source"] = new_source
            break

    # 1. Fix Random Forest Data Loading (Same as before)
    for cell in cells:
        if "Binary dataset not found" in "".join(cell.get("source", [])):
            # Check if already patched to avoid double patching
            if "X_activity[:, 0] += 2.0" in "".join(cell["source"]):
                print("Random Forest Data Loading already patched.")
                continue
                
            print("Patching Random Forest Data Loading...")
            new_source = []
            for line in cell["source"]:
                if "X, feature_names = generate_synthetic_data()" in line:
                    new_source.append("    # 1. Empty (Label 0)\n")
                    new_source.append("    X_empty, feature_names = generate_synthetic_data(n_samples=200)\n")
                    new_source.append("    y_empty = np.zeros(len(X_empty))\n")
                    new_source.append("    # 2. Activity (Label 1)\n")
                    new_source.append("    X_activity = X_empty.copy()\n")
                    new_source.append("    X_activity[:, 0] += 2.0 # Higher Variance\n")
                    new_source.append("    X_activity[:, 2] += 1.0 # Higher Velocity\n")
                    new_source.append("    y_activity = np.ones(len(X_activity))\n")
                    new_source.append("    X = np.concatenate([X_empty, X_activity])\n")
                    new_source.append("    y = np.concatenate([y_empty, y_activity])\n")
                elif "y = np.zeros(len(X))" in line:
                    continue # Skip
                else:
                    new_source.append(line)
            cell["source"] = new_source

    # 2. Fix CNN Data Loading (Same as before)
    # We need to find the cell with "class CSIDataset" and ensure imports are there too if needed, 
    # but global imports should handle it.
    
    # However, we need to make sure the CNN cell is robust.
    # The previous patch might have been overwritten if I re-ran create_notebook? 
    # No, I am editing the existing notebook.
    
    # Let's re-apply the CNN patch if it's not there.
    for cell in cells:
        if "try:" in "".join(cell.get("source", [])) and "full_dataset = CSIDataset" in "".join(cell.get("source", [])):
             if "X_synth_act = torch.randn" in "".join(cell["source"]):
                 print("CNN Data Loading already patched.")
                 continue
                 
             print("Patching CNN Data Loading...")
             # (Same code as before)
             new_code = """
# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Robust Data Loading
try:
    if Path("data/processed/windows").exists():
        full_dataset = CSIDataset("data/processed/windows")
        print(f"Loaded {len(full_dataset)} real samples.")
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    else:
        print("Real data not found. Using SYNTHETIC data.")
        train_dataset = None
        test_dataset = None

    # Generate Synthetic Data (Balanced)
    X_synth_empty, y_synth_empty = generate_synthetic_windows(n_samples=250)
    
    # Generate Synthetic Activity (Higher amplitude/noise)
    X_synth_act = torch.randn(250, 30, 256).float() * 2.0 + 1.0
    y_synth_act = torch.ones(250, dtype=torch.long)
    
    synth_dataset = torch.utils.data.TensorDataset(
        torch.cat([X_synth_empty, X_synth_act]), 
        torch.cat([y_synth_empty, y_synth_act])
    )
    
    if train_dataset is not None:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, synth_dataset])
    else:
        # Purely synthetic mode
        train_size = int(0.8 * len(synth_dataset))
        test_size = len(synth_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(synth_dataset, [train_size, test_size])
    
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

except Exception as e:
    print(f"Skipping CNN experiment: {e}")
"""
             cell["source"] = [line + "\n" for line in new_code.split("\n")]

    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("Notebook patched successfully.")

if __name__ == "__main__":
    fix_notebook()
