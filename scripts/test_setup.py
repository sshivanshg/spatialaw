#!/usr/bin/env python3
"""
Test Setup Script
Quick test to verify the baseline model setup is working
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    try:
        from src.models.baseline_model import BaselineSpatialModel
        print("✓ Baseline model imported")
    except ImportError as e:
        print(f"✗ Baseline model import failed: {e}")
        return False
    
    try:
        from src.data_collection.wifi_collector import WiFiCollector
        print("✓ WiFi collector imported")
    except ImportError as e:
        print(f"✗ WiFi collector import failed: {e}")
        return False
    
    try:
        from src.data_collection.csi_processor import CSIProcessor
        print("✓ CSI processor imported")
    except ImportError as e:
        print(f"✗ CSI processor import failed: {e}")
        return False
    
    try:
        from src.preprocessing.data_loader import CSIDataset
        print("✓ CSI dataset imported")
    except ImportError as e:
        print(f"✗ CSI dataset import failed: {e}")
        return False
    
    try:
        from src.training.trainer import Trainer
        print("✓ Trainer imported")
    except ImportError as e:
        print(f"✗ Trainer import failed: {e}")
        return False
    
    try:
        from src.evaluation.evaluator import Evaluator
        print("✓ Evaluator imported")
    except ImportError as e:
        print(f"✗ Evaluator import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        import torch
        from src.models.baseline_model import BaselineSpatialModel
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = BaselineSpatialModel(
            input_channels=2,
            num_subcarriers=64,
            num_antennas=3,
            latent_dim=128,
            output_channels=3,
            output_size=(64, 64)
        ).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 2, 3, 64).to(device)
        output = model(dummy_input)
        
        print(f"✓ Model created successfully")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csi_processor():
    """Test CSI processor."""
    print("\nTesting CSI processor...")
    
    try:
        from src.data_collection.csi_processor import CSIProcessor, generate_mock_csi
        
        processor = CSIProcessor(num_subcarriers=64, num_antennas=3)
        mock_csi = generate_mock_csi(num_samples=1, num_antennas=3, num_subcarriers=64)
        
        amplitude, phase = processor.process_csi(mock_csi[0])
        features = processor.extract_features(mock_csi[0])
        
        print(f"✓ CSI processor works")
        print(f"  Amplitude shape: {amplitude.shape}")
        print(f"  Phase shape: {phase.shape}")
        print(f"  Features: {list(features.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ CSI processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset creation."""
    print("\nTesting dataset creation...")
    
    try:
        from src.preprocessing.data_loader import CSIDataset
        from torch.utils.data import DataLoader
        
        dataset = CSIDataset(generate_mock=True, num_mock_samples=10)
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        sample = next(iter(data_loader))
        
        print(f"✓ Dataset created successfully")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  CSI shape: {sample['csi'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Baseline Model Setup Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("CSI Processor", test_csi_processor),
        ("Dataset", test_dataset),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! Setup is working correctly.")
        print("\nNext steps:")
        print("  1. Run: python scripts/quick_start.py")
        print("  2. Collect WiFi data: python scripts/collect_wifi_data.py")
        print("  3. Train model: python scripts/train_baseline.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Check Python version (requires Python 3.8+)")
        print("  3. Verify all files are in place")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

