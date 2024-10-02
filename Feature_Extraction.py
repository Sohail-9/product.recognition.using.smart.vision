import cv2

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        features.append((x, y, w, h))
    return features

features = extract_features('preprocessed_image.jpg')
print("Extracted Features:", features)