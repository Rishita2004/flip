# Futuristic Grid
- https://colab.research.google.com/drive/13L5c-T9yYxdGDVl3eD_fDUil7RvijYZO?usp=sharing

## Overview
This repository contains the implementation for classifying fruits as fresh or rotten using Deep Learning techniques. We initially established a baseline performance with a Convolutional Neural Network (CNN) and subsequently improved accuracy through Transfer Learning with EfficientNetB0.

## Dataset
We utilized the **Fruit Fresh and Rotten for Classification** dataset from Kaggle, which comprises **13,599 images** of apples, bananas, and oranges categorized into fresh and rotten classes.

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
from keras.preprocessing.image import img_to_array, load_img
import random, os

# Directories for testing
names = [fresh_apples_test_dir, fresh_banana_test_dir, fresh_oranges_test_dir,
         rotten_apples_test_dir, rotten_banana_test_dir, rotten_oranges_test_dir]

# Randomly selecting an image
name_rand = random.choice(names)
filename = os.listdir(name_rand)
sample = random.choice(filename)
fn = os.path.join(name_rand, sample)

# Load and display the image
img = load_img(fn, target_size=(150, 150))
plt.imshow(img)

# Preprocess for prediction
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

# Predict the class
classes = model.predict(images, batch_size=10)
print(classes)

# Interpret the prediction
prediction = ''
if classes == 1:
    prediction = 'fresh apple'
elif classes == 1:
    prediction = 'fresh banana'
elif classes == 1:
    prediction = 'fresh orange'
elif classes == 1:
    prediction = 'rotten apple'
elif classes == 1:
    prediction = 'rotten banana'
elif classes == 1:
    prediction = 'rotten orange'

print(prediction)         
```

## Future Enhancements
We plan to extend this solution into a web application integrated with IoT devices to enable real-time detection of fruit freshness for practical applications in grocery stores or warehouses. Additional enhancements may include:
- **Implementing a user-friendly interface** for easy interaction.
- **Adding support for more fruit and vegetable types** to broaden the classification capabilities.
- **Incorporating a feedback mechanism** to improve model accuracy over time.
- **Developing mobile applications** for on-the-go freshness checks.
