# SmartQC ( Quality Checking of Product using Smart Vision )

## Overview
This repository contains the implementation for classifying fruits as fresh or rotten using Deep Learning techniques. We initially established a baseline performance with a Convolutional Neural Network (CNN) and subsequently improved accuracy through Transfer Learning with EfficientNetB0. 
Along with the implementation of OCR to extract details from product image data that were web scraped from the Flipkart Supermart website. The technologies that were used for OCR implementation are feature extraction techniques that have edge and contour detection and were used to analyse product packaging, and object detection, which involved a pretrained model (YOLO 5) and was used to detect product shape and text from images. Then OCR to extract and recognize text from product image, focusing on brand name, product name, and pack size. 

## Dataset
We utilized the **Fruit Fresh and Rotten for Classification** dataset from Kaggle (https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification), which comprises **13,599 images** of apples, bananas, and oranges categorized into fresh and rotten classes.

Web scraped data from Flipkart Supermart Website for OCR implementation (https://drive.google.com/drive/folders/1J3tSC5bZz6Sj6dxskvIrpn5WogmDLWQH?usp=sharing) .

We create our own dataset to implement brand detection & expiry date extraction 
* Annotation of unlabelled image data using Roboflow .
* Execution of YOLOv8 to extract details such as BRAND NAME ,PRODUCT NAME , MRP , EXPIRY DATE 
(https://drive.google.com/drive/folders/1KL0Qvyof8BUAYXex1m3T-ldX-_QlZrYG?usp=sharing)

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
- **YOLO 5**
- **pytesseract**

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
def predict_image(image_path):
    model.eval()  # Set the model to evaluation mode

    # Open the image
    img = Image.open(image_path).convert("RGB")

    # Apply the transformations
    transformed = manual_transform(img).to(device)

    # Inference
    with torch.inference_mode():
        logits = model(transformed.unsqueeze(dim=0))  # Add batch dimension
        pred = torch.softmax(logits, dim=-1)

    # Get the prediction and confidence
    predicted_class = class_names[pred.argmax(dim=-1).item()]
    confidence = pred.max().item()

    return predicted_class, confidence, img

# Example of running inference on a single image
image_path = "/content/PATH_TO_YOUR_IMAGE.jpg"  # Provide the path to the image here

# Get prediction and the image
predicted_class, confidence, img = predict_image(image_path)

# Plotting the image with prediction
plt.imshow(img)
plt.title(f"Prediction: {predicted_class} | Confidence: {confidence:.3f}")
plt.axis('off')  # Hide the axis
plt.show()   
```

## Future Enhancements
We plan to extend this solution into a web application integrated with IoT devices to enable real-time detection of fruit freshness for practical applications in grocery stores or warehouses. Additional enhancements may include:
- **Implementing a user-friendly interface** for easy interaction.
- **Adding support for more fruit and vegetable types** to broaden the classification capabilities.
- **Incorporating a feedback mechanism** to improve model accuracy over time.
- **Developing mobile applications** for on-the-go freshness checks.
