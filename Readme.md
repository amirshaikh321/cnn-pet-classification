# 🐾 DeepVision Pet Classifier: Dog vs. Cat Classifier

A high-performance **Convolutional Neural Network (CNN)** built from scratch using **PyTorch** to solve the binary classification task of identifying dogs and cats.

---

## 📖 Overview
This project demonstrates a complete deep learning workflow—from raw image data to a trained inference model. By leveraging PyTorch's dynamic computational graph, the model learns to identify key spatial features (like ear shapes, snout length, and fur texture) to distinguish between felines and canines.



## ✨ Key Features
* **Built with PyTorch:** Utilizes `torch` for tensor computations and `torchvision` for pre-built datasets and transformations.
* **Data Augmentation:** Robust preprocessing including random horizontal flips, rotations, and color jittering to prevent overfitting.
* **Optimized Performance:** Supports **CUDA GPU** acceleration for significantly faster training.
* **Modular Design:** Separated scripts for data loading (`utils.py`), model architecture (`model.py`), and training (`engine.py`).
* **Detailed Metrics:** Outputs training/validation loss and accuracy curves for performance monitoring.



## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Computer Vision:** Torchvision, OpenCV
* **Data Analysis/Visulization:** NumPy, Matplotlib, PIL

## 📂 Project Structure
```text
├── data/               # Dataset directory (train/test)
├── models/             # Saved model weights (.pth files)
├── src/                
├── app.py              # Data loaders and transforms
├── utils.py            # train, test, model_evaluate functions
├── model.py            # CNN architecture definition
├── predict.py          # Prediction script
└── model.ipynb         # Model train test evaluation Notebook
├── requirements.txt    # Project dependencies
└── README.md