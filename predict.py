import torch
from model import BetterCNN
import os
model = BetterCNN()
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the model file
model_path = os.path.join(current_dir, "pet_cnn.pth")

# Load using the full path
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
