# config.py
# All project settings in one place — change here, applies everywhere

# Animals we detect
CLASS_NAMES = ['bear', 'deer', 'elephant', 'gorilla', 'leopard', 'lion', 'tiger', 'zebra']
NUM_CLASSES = len(CLASS_NAMES)

# Image sizes
TEACHER_IMG_SIZE = 160
STUDENT_IMG_SIZE = 96

# Training
BATCH_SIZE = 16
TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 60
DISTILL_EPOCHS = 80

# Distillation
TEMPERATURE = 5.0
ALPHA = 0.7

# Dataset
TARGET_IMAGES_PER_ANIMAL = 1000
TRAIN_RATIO = 0.85

# Paths
DATASET_DIR = "dataset"
RAW_DIR = "raw_downloads"
TEACHER_MODEL_PATH = "teacher_model.keras"
STUDENT_MODEL_PATH = "student_model.keras"
CLASS_NAMES_FILE = "class_names.txt"

# ESP32-CAM
ESP32_IMG_SIZE = 96
TFLITE_MODEL_PATH = "animal_model.tflite"
C_HEADER_PATH = "animal_model.h"