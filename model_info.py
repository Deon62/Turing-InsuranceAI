import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 


model_path = "car_damage_resnet50.h5"
model = load_model(model_path)
model.summary()

for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")
    
input_shape = model.input_shape
print(f"Input shape: {input_shape}")
output_shape = model.output_shape
print(f"Output shape: {output_shape}")