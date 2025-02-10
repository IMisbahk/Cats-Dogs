from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('catdog.keras')

def preprocess_image(image_path, target_size=(64,64)):
    img = Image.open(image_path).resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

imagepath = 'murphy.jpg'
img = preprocess_image(imagepath)
prediction = model.predict(img)

if prediction[0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")