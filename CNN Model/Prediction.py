from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('product_classification_model.h5')

def classify_product(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    return 'Defective' if prediction < 0.5 else 'Non-Defective'

result = classify_product('preprocessed_image.jpg')
print("Classification Result:", result)
