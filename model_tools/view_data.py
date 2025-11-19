#!/usr/bin/env python
# coding: utf-8

# # Read features.npy
# 
# Simple notebook to read and view the features.npy file.
# 

# In[24]:


import numpy as np
from pathlib import Path

# Find data directory
root = Path().resolve().parent if (Path().resolve().parent / "data").exists() else Path().resolve()


# In[25]:


# Load features.npy
features = np.load(root / "data/processed/binary/features.npy")
print("Features shape:", features.shape)
print("\nAll Features Data (no limits):")
np.set_printoptions(threshold=np.inf, edgeitems=0, linewidth=np.inf)
print(features)


# In[26]:


# Print first few samples
print(features)


# In[27]:


# Print all samples with index numbers
print("All samples with indices:")
for i in range(len(features)):
    print(f"\nSample {i}:")
    print(features[i])


# # Read and View .npy Files
# 
# Simple notebook to read and print data from .npy files.
# 

# In[28]:


import numpy as np
from pathlib import Path

# Find data directory
root = Path().resolve().parent if (Path().resolve().parent / "data").exists() else Path().resolve()


# ## Read Binary Dataset Features
# 

# In[29]:


# Load and print features
features = np.load(root / "data/processed/binary/features.npy")
print("Features shape:", features.shape)
print("\nFeatures data:")
print(features)


# In[30]:


# Print first few samples
print("\nFirst 5 samples:")
print(features[:5])


# In[31]:


## Read Processed Windows


# 
# 

# In[32]:


# Load and print a window
window = np.load(root / "data/processed/windows/window_000000.npy")
print("Window shape:", window.shape)
print("\nWindow data:")
print(window)


# In[33]:


# Print first few rows and columns
print("\nFirst 10 rows, first 20 columns:")
print(window[:10, :20])


# In[34]:


# Load another window
window2 = np.load(root / "data/processed/windows/window_000001.npy")
print("Second window shape:", window2.shape)
print("\nSecond window data:")
print(window2)

