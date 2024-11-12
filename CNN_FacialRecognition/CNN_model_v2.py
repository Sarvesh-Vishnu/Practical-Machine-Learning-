# face_recognition_lfw.py

"""
Face Recognition with LFW Dataset Using Deep Learning

This project implements a robust face recognition system using the Labeled Faces in the Wild (LFW) dataset
and a deep learning model based on ResNet50. The application leverages high-quality feature embeddings,
real-time data augmentation, and cosine similarity for face matching.

Project Structure:
1. Dataset Preparation and Preprocessing: Load and preprocess images from the LFW dataset with data augmentation.
2. Model Architecture: Fine-tune a ResNet50 model to create unique embeddings for each face.
3. Embeddings and Matching: Extract feature embeddings and perform face matching using cosine similarity.
4. Model Evaluation: Use t-SNE for visualization and calculate evaluation metrics.
"""

# Imports
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tensorflow.keras.metrics import AUC, Precision, Recall

# Add configuration class for better parameter management
class Config:
    def __init__(self):
        self.dataset_path = 'path/to/LFW-dataset'
        self.batch_size = 32
        self.target_size = (224, 224)
        self.epochs = 10
        self.learning_rate = 0.0001
        self.embedding_size = 512

config = Config()

# Improved Data Generator with preprocessing function
def preprocess_image(image):
    return cv2.resize(image, config.target_size) / 255.0

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=20,  # Increased for better augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,  # Added zoom augmentation
    horizontal_flip=True,
    validation_split=0.2
)

# Training generator
train_generator = datagen.flow_from_directory(
    directory=config.dataset_path,
    target_size=config.target_size,
    batch_size=config.batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation generator
validation_generator = datagen.flow_from_directory(
    directory=config.dataset_path,
    target_size=config.target_size,
    batch_size=config.batch_size,
    class_mode='categorical',
    subset='validation'
)

# Improved Model Architecture with dropout for better generalization
def build_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*config.target_size, 3))
    
    # Freeze only the first 100 layers instead of all
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(config.embedding_size, activation='relu')(x)
    x = Dropout(0.5)(x)  # Added dropout
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=outputs)

# Create model using the builder function
model = build_model(train_generator.num_classes)

# Add learning rate scheduler for better training
def lr_schedule(epoch):
    return config.learning_rate * (0.1 ** (epoch // 3))

lr_scheduler = LearningRateScheduler(lr_schedule)

# Improved model compilation with additional metrics
model.compile(
    optimizer=Adam(learning_rate=config.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(), Precision(), Recall()]
)

# Enhanced training with callbacks
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config.epochs,
    callbacks=[
        lr_scheduler,
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ]
)

# Save model
model.save("lfw_face_recognition_model.h5")

# Embedding Extraction
embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Generate and store embeddings for known faces
known_embeddings = {}
for class_label, class_name in train_generator.class_indices.items():
    class_path = os.path.join(config.dataset_path, class_name)
    class_images = os.listdir(class_path)
    embeddings = []
    for img_name in class_images:
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0)
        embedding = embedding_model.predict(img)[0]
        embeddings.append(embedding)
    known_embeddings[class_name] = np.mean(embeddings, axis=0)  # Average embedding for the class

# Improved matching function with threshold
def match_face(new_image_path, threshold=0.6):
    new_image = preprocess_image(cv2.imread(new_image_path))
    new_embedding = embedding_model.predict(np.expand_dims(new_image, axis=0), verbose=0)[0]
    
    similarities = {name: cosine_similarity([new_embedding], [emb])[0][0] 
                   for name, emb in known_embeddings.items()}
    match_name = max(similarities, key=similarities.get)
    confidence = similarities[match_name]
    
    return (match_name, confidence) if confidence >= threshold else ("Unknown", confidence)

# Example Usage
# result_name, confidence = match_face('path/to/test_image.jpg')
# print(f"Matched with: {result_name} (Confidence: {confidence:.2f})")

# Improved visualization function with better plotting
def plot_embeddings(perplexity=30, figsize=(12, 8)):
    embeddings = np.array(list(known_embeddings.values()))
    labels = list(known_embeddings.keys())
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1)
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                         c=np.arange(len(labels)), cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Embedding Visualization")
    plt.tight_layout()
    plt.show()

# Uncomment to visualize embeddings after training
# plot_embeddings()

# Improved embedding extraction with batch processing
def extract_embeddings(image_paths, batch_size=32):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = np.array([preprocess_image(cv2.imread(path)) for path in batch_paths])
        batch_embeddings = embedding_model.predict(batch_images, verbose=0)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)
