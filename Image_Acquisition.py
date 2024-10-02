import cv2

def capture_image(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_image.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()

capture_image()
