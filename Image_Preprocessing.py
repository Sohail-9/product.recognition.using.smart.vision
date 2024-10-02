import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    return blurred

preprocessed_image = preprocess_image('captured_image.jpg')
cv2.imwrite('preprocessed_image.jpg', preprocessed_image)
