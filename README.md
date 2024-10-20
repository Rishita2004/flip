# Futuristic Grid

## Overview
This repository contains the implementation for classifying fruits as fresh or rotten using Deep Learning techniques. We initially established a baseline performance with a Convolutional Neural Network (CNN) and subsequently improved accuracy through Transfer Learning with EfficientNetB0.

## Dataset
We utilized the **Fruit Fresh and Rotten for Classification** dataset from Kaggle (https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification), which comprises **13,599 images** of apples, bananas, and oranges categorized into fresh and rotten classes.

### Dataset Structure
| Dataset | Directories      | Files |
|---------|------------------|-------|
| Train   | freshapples      | 1693  |
|         | freshbanana      | 1581  |
|         | freshoranges     | 1466  |
|         | rottenapples     | 2342  |
|         | rottenbanana     | 2224  |
|         | rottenoranges    | 1595  |
| Test    | freshapples      | 395   |
|         | freshbanana      | 381   |
|         | freshoranges     | 388   |
|         | rottenapples     | 601   |
|         | rottenbanana     | 530   |
|         | rottenoranges    | 403   |

## Technologies Used
- **Convolutional Neural Networks (CNN)**
- **EfficientNetB0 (Transfer Learning)**
- **TensorFlow**
- **Keras**
- **OpenCV**

## Algorithm and Process
- - https://colab.research.google.com/drive/13L5c-T9yYxdGDVl3eD_fDUil7RvijYZO?usp=sharing
1. **Data Preparation**: Load and inspect the dataset, randomly select a sample fruit image for preview.
2. **Image Resizing**: Resize all images to (150, 150) pixels and normalize pixel values.
3. **Modeling**: Utilize EfficientNetB0 with Transfer Learning, leveraging pre-trained weights.
4. **Model Training**: Train the model while tracking accuracy and loss metrics over epochs.
5. **Evaluation**: Evaluate model performance and visualize results.
6. **Prediction**: Test the model by predicting classes of random images from the test dataset.

### Model Performance
| Model           | Total Parameters | Loss   | Accuracy | Optimizer | Loss Metric            |
|-----------------|------------------|--------|----------|-----------|------------------------|
| EfficientNetB0  | 5,330,571        | 0.0145 | 98.12%   | Adam      | Categorical CrossEntropy|

## Code for Prediction
```python
# Example Usage
# To make a prediction using the trained EfficientNet-B0 model:

import torch
from PIL import Image

# Load the pre-trained model and ensure it's in evaluation mode
model.load_state_dict(torch.load("/path_to_model/effnetb0_freshvisionv0_10_epochs.pt", map_location="cpu"))
model.eval()

# Define function to predict class
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transformed_img = manual_transform(img).unsqueeze(0).to(device)  # Add batch dimension
    with torch.inference_mode():
        logits = model(transformed_img)
        probs = torch.softmax(logits, dim=-1)
        predicted_class = class_names[probs.argmax().item()]
        confidence = probs.max().item()

    return f"Prediction: {predicted_class} | Confidence: {confidence:.3f}"

# Example image path
image_path = "/path_to_image/banana.jpg"
print(predict_image(image_path))     
```

## Future Enhancements
We plan to extend this solution into a web application integrated with IoT devices to enable real-time detection of fruit freshness for practical applications in grocery stores or warehouses. Additional enhancements may include:
- **Implementing a user-friendly interface** for easy interaction.
- **Adding support for more fruit and vegetable types** to broaden the classification capabilities.
- **Incorporating a feedback mechanism** to improve model accuracy over time.
- **Developing mobile applications** for on-the-go freshness checks.
