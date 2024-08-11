import cv2
from PIL import Image
import numpy as np
from util import get_limits

# Example color dictionary with BGR values and corresponding color names
color_dict = {
    'yellow': [0, 255, 255],
    'blue': [255, 0, 0],
    'green': [0, 255, 0],
    'red': [0, 0, 255],
    'purple': [128, 0, 128],
    'brown': [42, 42, 165],
}

def frame_difference(frame1, frame2):
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame1, gray_frame2)
    _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return np.sum(diff_thresh)

def detect_prominent_color(hsv_image):
    max_area = 0
    detected_color = None

    for color_name, color_bgr in color_dict.items():
        lowerLimit, upperLimit = get_limits(color=color_bgr)
        mask = cv2.inRange(hsv_image, lowerLimit, upperLimit)
        area = np.sum(mask > 0)

        if area > max_area:
            max_area = area
            detected_color = color_name

    return detected_color

cap = cv2.VideoCapture(1)

# Capture the reference frame (background)
ret, reference_frame = cap.read()

while True:
    ret, frame = cap.read()

    # Calculate the difference between the current frame and the reference frame
    diff = frame_difference(reference_frame, frame)

    # Start detection only if the difference exceeds a threshold
    if diff > 50000:  # Adjust the threshold based on your environment
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_color = detect_prominent_color(hsvImage)

        if detected_color:
            # Get BGR value for detected color
            color_bgr = color_dict[detected_color]

            # Create a mask for the detected color
            lowerLimit, upperLimit = get_limits(color=color_bgr)
            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

            mask_ = Image.fromarray(mask)
            bbox = mask_.getbbox()

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 5)

                # Display the color name at the top-left corner of the rectangle
                cv2.putText(frame, detected_color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
