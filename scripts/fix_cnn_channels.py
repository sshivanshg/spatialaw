import json

def fix_cnn_channels():
    notebook_path = "Spatial_Awareness_Project.ipynb"
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    cells = notebook['cells']
    
    for cell in cells:
        if "class SimpleCNN(nn.Module):" in "".join(cell.get("source", [])):
            print("Patching SimpleCNN input channels...")
            new_source = []
            for line in cell["source"]:
                if "self.conv1 = nn.Conv1d(in_channels=30" in line:
                    new_source.append(line.replace("in_channels=30", "in_channels=60"))
                elif "# Input: (Batch, 30, 256)" in line:
                    new_source.append(line.replace("30", "60"))
                else:
                    new_source.append(line)
            cell["source"] = new_source
            
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("Notebook patched: SimpleCNN now accepts 60 channels.")

if __name__ == "__main__":
    fix_cnn_channels()
