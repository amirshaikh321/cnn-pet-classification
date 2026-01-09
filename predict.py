import torch
from model import BetterCNN

model = BetterCNN()
model.load_state_dict(torch.load("pet_cnn.pth", map_location="cpu"))
model.eval()
