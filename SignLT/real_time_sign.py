import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model and label encoder
model = tf.keras.models.load_model("sign_language_model.h5")
label_encoder = np.load("label_encoder.npy", allow_pickle=True)

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

def draw_speech_bubble(frame, text, position, bubble_color=(255, 255, 255), text_color=(0, 0, 0)):
    """Draws an automatically sized speech bubble with extra-large text."""
    x, y = position
    font_scale = 3.5
    thickness = 8

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    padding = 40
    w = text_width + padding
    h = text_height + padding

    bubble_x = x - w // 2
    bubble_y = y - h - 50
    bubble_x = max(20, min(bubble_x, frame.shape[1] - w - 20))
    bubble_y = max(20, bubble_y)

    cv2.rectangle(frame, (bubble_x, bubble_y), (bubble_x + w, bubble_y + h), bubble_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (bubble_x, bubble_y), (bubble_x + w, bubble_y + h), (0, 0, 0), 3, cv2.LINE_AA)

    tail_x = bubble_x + w // 2
    tail_y = bubble_y + h
    cv2.circle(frame, (tail_x, tail_y + 15), 15, bubble_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (tail_x, tail_y + 15), 15, (0, 0, 0), 3, cv2.LINE_AA)

    text_x = bubble_x + 20
    text_y = bubble_y + h - 20
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            landmarks = np.array(landmarks).flatten().reshape(1, -1, 1)
            prediction = model.predict(landmarks)
            gesture_text = label_encoder[np.argmax(prediction)]

    if gesture_text:
        h, w, _ = frame.shape
        draw_speech_bubble(frame, gesture_text, (w // 2, 150))

    cv2.imshow("Real-Time Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
