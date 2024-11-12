# Face Recognition with LFW Dataset Using Deep Learning

This project implements a face recognition system using the **Labeled Faces in the Wild (LFW)** dataset and a deep learning model based on **ResNet50**. The system generates unique face embeddings for individuals and performs face matching using cosine similarity. This approach demonstrates high accuracy and robustness, making it suitable for face recognition tasks.

## Project Overview

1. **Dataset Preparation and Preprocessing**: Load and preprocess images with data augmentation for improved generalization.
2. **Model Architecture**: Fine-tune a ResNet50 model to extract facial feature embeddings.
3. **Embeddings and Matching**: Generate embeddings and perform face matching using cosine similarity.
4. **Model Evaluation**: Use t-SNE for visualizing embeddings and analyze clustering for similar faces.

## Features

- **Data Augmentation**: Applies transformations like rotation, shifting, and flipping to enhance model generalization.
- **Fine-tuned ResNet50**: Uses a pre-trained ResNet50 model for high-quality feature extraction.
- **Face Embedding and Matching**: Generates embeddings for known faces and matches new faces based on cosine similarity.
- **t-SNE Visualization**: Provides visualization of embeddings to analyze how well the model clusters similar faces.

## Dataset

We use the [Labeled Faces in the Wild (LFW) dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) for training and testing. The LFW dataset contains over 13,000 labeled images of faces collected from the internet, providing diverse variations in pose, lighting, and expressions.

- **Number of Images**: 13,233
- **Classes**: Each class corresponds to an individual

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- Matplotlib

Install dependencies with:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib

