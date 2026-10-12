import torch
import torchvision
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from PIL import Image

print("Environment verification successful!")
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Streamlit:", st.__version__)
print("CUDA available:", torch.cuda.is_available())