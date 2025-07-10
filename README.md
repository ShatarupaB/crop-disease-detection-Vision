# Crop Disease Detection using YOLOv11 and CNN Models

This project implements a hybrid deep learning approach to detect and classify crop diseases using image data. Leveraging the YOLOv11 architecture, along with CNN-based models such as MobileNetV2, EfficientNetB0, and ResNet50, the system was trained and evaluated on the PlantVillage dataset. It aims to contribute to the field of precision agriculture by offering an effective solution for early disease detection in crops like Tomato, Bell Pepper, and Potato.

---

## Objective

To design an efficient crop disease detection system using:
- **YOLOv11** for object detection and classification
- Comparative evaluation with **MobileNetV2**, **EfficientNetB0**, and **ResNet50**
- Augmented and balanced dataset based on **PlantVillage**
- High performance across multiple classes of crop diseases

---

## Models Used

| Model         | Validation Accuracy | Test Accuracy |
|---------------|---------------------|----------------|
| YOLOv11       | 99.9%               | 100%           |
| ResNet50      | 97.75%              | 98.09%         |
| MobileNetV2   | 97.62%              | 98.09%         |
| EfficientNetB0| 97.73%              | 97.61%         |

---

## Dataset

- **PlantVillage** dataset
- Diseases classified:
  - Late Blight
  - Early Blight
  - Bacterial Spot
  - Leaf Spot
  - Curl Virus
  - Mosaic Virus
- **Preprocessing**:
  - Image augmentation (flip, rotate, resize)
  - Class balancing: ~2000 images per class
  - Split: 70% train / 20% validation / 10% test

---

## Technologies Used

- Python
- Jupyter Notebook
- YOLOv11 (custom-trained)
- PyTorch, TensorFlow/Keras (for comparative models)
- Matplotlib, Seaborn

---

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Training/Validation Loss Curves


