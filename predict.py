import torch
from model import PetCNN

model = PetCNN()
model.load_state_dict(torch.load("pet_cnn.pth", map_location="cpu"))
model.eval()
