# Deep Fake Detection

This repository contains the code and resources for a Deep Fake detection project. The project leverages convolutional neural networks (CNNs) to classify real and fake videos, aiming to detect deep fakes with high accuracy.

## Project Overview

Deep fake videos are synthetic media where the appearance of a person is replaced with another's likeness. This project focuses on developing a system to automatically detect these manipulations using video frames as input.

## Features

- **Video Frame Extraction**: Load videos and extract frames for analysis.
- **Data Preprocessing**: Resize and normalize video frames for input into the CNN.
- **CNN Model**: Custom convolutional neural network designed for binary classification (real or fake).
- **Training and Validation**: Train the model using a split dataset and evaluate its performance.
- **Prediction and Reporting**: Predict video authenticity and generate detailed reports.
- **Model Persistence**: Save the trained model for future use and load it for predictions.

## Dataset

The project uses the **Celeb-DF** dataset, available on Kaggle. This dataset contains both real and fake videos of celebrities, making it ideal for training and validating deep learning models.

### Dataset Structure

- **Celeb-DF-v1**: Initial version of the dataset.
- **Celeb-DF-v2**: Enhanced version with more sophisticated deep fakes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd deepfake-detection
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset**: Download the dataset from Kaggle and organize it as required.
2. **Run the Training Script**: Execute the `sih_try.py` script to train the model.
3. **Evaluate and Predict**: Use the trained model to evaluate new videos and generate reports.

## Files

- **sih_try.py**: Main script containing the data loading, preprocessing, model training, testing and prediction pipeline and saved model for future predictions.

## Model Architecture

The CNN model includes:
- **Convolutional Layers**: Extract spatial features from video frames.
- **MaxPooling Layers**: Downsample feature maps.
- **Dense Layers**: Fully connected layers for classification.
- **Sigmoid Output Layer**: Outputs the probability of a frame being fake.

## How It Works

1. **Frame Extraction**: Extract frames from videos for analysis.
2. **Preprocessing**: Resize frames to 224x224 pixels and normalize pixel values.
3. **Model Training**: Train the CNN model on the processed frames.
4. **Prediction**: Classify whether a video is real or fake.
5. **Report Generation**: Generate a detailed report of the findings.

## Who Can Benefit

- **Media Forensics**: Detect and prevent the spread of fake media.
- **Developers and Researchers**: Explore applications of deep learning in media authenticity.
- **Educational Institutions**: A hands-on project for learning about deep learning, computer vision, and ethical AI.

## Use Cases

- **Social Media Platforms**: Automatically detect and flag deep fake content.
- **News Organizations**: Verify the authenticity of video content.
- **Legal Entities**: Provide evidence in cases involving media manipulation.

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for the Celeb-DF dataset.

