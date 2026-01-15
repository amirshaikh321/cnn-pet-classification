import torch
import os
from torchvision import transforms
from model import BetterCNN  # Ensures model structure matches

# Define the image transformation (Must match training!)
# Assuming you trained on 64x64 or 128x128. Adjust resize if needed.
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Change to 64 if you trained on 64
    transforms.ToTensor(),
])

def predict_image(image_obj):
    # 1. Initialize Model
    model = BetterCNN(num_classes=2)
    
    # 2. Construct dynamic path to the CORRECT filename
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # CORRECTED: You said the file is 'pet_classifier.pth', not 'pet_cnn.pth'
    model_path = os.path.join(current_dir, "models", "pet_classifier.pth") 

    # 3. Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except FileNotFoundError:
        return f"Error: Model file not found at {model_path}"
    
    model.eval()

    # 4. Preprocess the Image
    # image_obj comes from app.py as a PIL Image
    input_tensor = transform(image_obj).unsqueeze(0)  # Add batch dimension -> [1, 3, 128, 128]

    # 5. Predict
    with torch.no_grad():
        output = model(input_tensor)
        # Your model outputs raw logits for 2 classes [score_cat, score_dog]
        _, predicted_idx = torch.max(output, 1)
        
        # 6. Map index to label
        # Usually: 0 = Cat, 1 = Dog (Depends on your training class_to_idx)
        labels = {0: "Cat 🐱", 1: "Dog 🐶"}
        return labels[predicted_idx.item()]