# Playing Card Detection & Recognition using YOLOv8

This project implements a complete end-to-end **playing card detector and classifier** using the YOLOv8 object detection framework.  
The model is trained to identify **all 52 standard playing cards** (e.g., “4h”, “Kd”, “10s”) and draw labeled bounding boxes around each card in an image.

This work was completed as the final project for **CS 4337 – Introduction to Machine Learning for Computer Vision**.

---

## Features

- Detects and classifies all **52 card classes**
- Works on rotated, skewed, and overlapping cards
- Fast inference using GPU acceleration
- Training script (`train_cards.py`)
- Prediction script (`predict_card.py`)
- Supports directory-based and single-image inference
- Includes trained model weights (`best.pt`)

---

## Dataset

**Dataset used:**  
Kaggle – _Playing Cards Object Detection Dataset_  
https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset

Includes:

- all **52 card classes**
- bounding-box annotations
- rotated/angled cards
- synthetic backgrounds
- separate train/validation/test splits

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/playing-card-detector.git
cd playing-card-detector
```

### 2. Create a virtual environment (Windows)

```powershell
python -m venv cards_env
cards_env\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Training the Model

To train the YOLOv8 model on your dataset:

```powershell
python train_cards.py
```

This will create weights here:

```
runs_cards/cards_detector2/weights/best.pt
```

---

## Running Inference

### Predict on the test directory

```powershell
python predict_card.py
```

### Predict on a specific image

Inside `predict_card.py` change:

```python
source = "path/to/image.jpg"
```

Predictions will be saved to:

```
runs_cards/predictions/
```

---

## Results

- Accurate detection of rotated and angled cards
- Successful recognition of all 52 classes
- Strong generalization to real-world cluttered backgrounds
- Average inference time: **5–20 ms per image**

---

## References

- Ultralytics YOLOv8 Docs
- Andy8744 Playing Cards Dataset
- CS 4337 Course Materials
