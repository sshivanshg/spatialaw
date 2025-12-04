import pandas as pd
from pathlib import Path

# Paths
WINDOWS_DIR = Path("data/processed/windows")
BINARY_DIR = Path("data/processed/windows_binary")
LABELS_PATH = WINDOWS_DIR / "labels.csv"
SYNTH_PATH = BINARY_DIR / "synthetic_windows.csv"

def merge_labels():
    print("Loading datasets...")
    df_main = pd.read_csv(LABELS_PATH)
    df_synth = pd.read_csv(SYNTH_PATH)
    
    print(f"Main dataset: {len(df_main)} rows")
    print(f"Synthetic dataset: {len(df_synth)} rows")
    
    # Prepare synthetic dataframe to match main schema
    # Main columns: activity_id, activity_name, auto_label, label, source_recording, window_file
    
    # Map synthetic columns
    df_synth_mapped = pd.DataFrame()
    df_synth_mapped['window_file'] = df_synth['window_file']
    df_synth_mapped['label'] = 0  # Force label 0 for Empty/No Activity
    df_synth_mapped['activity_id'] = 0 # 0 usually reserved for No Activity
    df_synth_mapped['activity_name'] = 'empty_room'
    df_synth_mapped['auto_label'] = False
    df_synth_mapped['source_recording'] = 'synthetic'
    
    # Append
    df_merged = pd.concat([df_main, df_synth_mapped], ignore_index=True)
    
    # Save
    df_merged.to_csv(LABELS_PATH, index=False)
    print(f"Merged dataset saved to {LABELS_PATH}")
    print(f"Total rows: {len(df_merged)}")
    print(f"Class distribution:\n{df_merged['label'].value_counts().sort_index()}")

if __name__ == "__main__":
    merge_labels()
