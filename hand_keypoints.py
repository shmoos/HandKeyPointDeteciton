#imports the libraries 
import cv2  
import mediapipe as mp
import time

# shortcuts using mediuapipe modules 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create detector (up to 2 hands)
hands = mp_hands.Hands(
    static_image_mode=False, # not static, so showing real time updating 
    max_num_hands=2, # sets max num of hands to detect 
    min_detection_confidence=0.7, # threshold 
    min_tracking_confidence=0.5 # stabilizing landmarks 
)

# detects if the hand is open 
def is_hand_open(hand_landmarks):
    tips = [8, 12, 16, 20]  # fingertips for index, middle, ring, pinky
    open_fingers = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_fingers += 1
    return open_fingers >= 3 # if three or more fingers are open then  hand is open, vice versa 

# Start video
cap = cv2.VideoCapture(0)  # intializiing webcam 
cap.set(3, 1920) # sets rez 
cap.set(4, 1080) # sets rez 
prev_time = 0

while cap.isOpened():
    success, frame = cap.read() # reads frame by frame of webcam 
    if not success:
        continue

    # Depicts left hand and right hand 
    frame = cv2.flip(frame, 1)
    


    # Convert the frame to RGB (MediaPipe expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RBG for mp 

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # Draw landmarks and detect gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Get bounding box coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            h, w, _ = frame.shape
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            # Gesture detection
            if is_hand_open(hand_landmarks):
                cv2.putText(frame, "Open Hand", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Closed Hand", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Keypoints", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
