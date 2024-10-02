import pytesseract

def extract_text(image_path):
    text = pytesseract.image_to_string(image_path)
    return text

text = extract_text('preprocessed_image.jpg')
print("Extracted Text:", text)
