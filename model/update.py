import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('imageclassifier.h5')
model.save('imageclassifier_keras3.h5', save_format='h5')