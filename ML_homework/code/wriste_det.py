import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_wrist_palm(image_path, output_path):
    # Load the input image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("No hands detected.")
        return

    for hand_landmarks in results.multi_hand_landmarks:
        # Extract the wrist and palm landmarks
        palm_landmarks = [mp_hands.HandLandmark.WRIST, 
                          mp_hands.HandLandmark.THUMB_CMC,
                          mp_hands.HandLandmark.THUMB_MCP,
                          mp_hands.HandLandmark.INDEX_FINGER_MCP,
                          mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        # Get the coordinates of the palm landmarks
        h, w, _ = image.shape
        coords = [(int(landmark.x * w), int(landmark.y * h)) for i, landmark in enumerate(hand_landmarks.landmark) if i in palm_landmarks]

        # Calculate the bounding box
        x_min = min([coord[0] for coord in coords])
        x_max = max([coord[0] for coord in coords])
        y_min = min([coord[1] for coord in coords])
        y_max = max([coord[1] for coord in coords])

        # Add padding around the bounding box
        padding = 20
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)

        # Crop and save the wrist and palm region
        cropped = image[y_min:y_max, x_min:x_max]
        cv2.imwrite(output_path, cropped)

        print(f"Cropped wrist and palm saved to {output_path}")

# Example usage
detect_and_crop_wrist_palm("../man_images/Nman.jpg", "output.jpg")
