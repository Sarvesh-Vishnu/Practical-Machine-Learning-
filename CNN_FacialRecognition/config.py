# config.py

# Path to the LFW dataset directory
DATASET_PATH = 'data/lfw-deepfunneled'  # Update this path based on your dataset location

# Training parameters
BATCH_SIZE = 32              # Number of samples per batch
TARGET_SIZE = (224, 224)     # Target image size for the model
EPOCHS = 10                  # Number of training epochs
LEARNING_RATE = 0.0001       # Learning rate for the Adam optimizer

# Model and embedding settings
EMBEDDING_LAYER_SIZE = 512   # Number of units in the dense layer before the output
OUTPUT_LAYER_ACTIVATION = 'softmax'  # Activation for the output layer in classification
