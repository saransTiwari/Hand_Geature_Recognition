import cv2
import mediapipe as mp
import numpy as np

# Define a dictionary mapping gesture labels to integers
gesture_dict = {0: 'up-left', 1: 'up-right', 2: 'down-left', 3: 'down-right'}


# Initialize the MediaPipe Hands model
mp_hands = mp.solutions.hands.Hands()

# Initialize the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB for use with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect hands in the RGB frame using MediaPipe
    results = mp_hands.process(frame_rgb)
    # If hands are detected, draw landmarks on the frame and recognize the gesture
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            # Extract the landmarks for the hand
            hand_landmarks_list = []
            for landmark in hand_landmarks.landmark:
                hand_landmarks_list.append(landmark.x)
                hand_landmarks_list.append(landmark.y)
                hand_landmarks_list.append(landmark.z)
            # Map the landmarks to a gesture based on the position of the thumb and index finger
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
            index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
            if thumb_tip[1] < index_tip[1] and thumb_tip[0] > index_tip[0]:
                gesture_name = gesture_dict[2]  # down
            elif thumb_tip[1] > index_tip[1] and thumb_tip[0] < index_tip[0]:
                gesture_name = gesture_dict[1]
            elif thumb_tip[1] > index_tip[1] and thumb_tip[0] > index_tip[0]:
                gesture_name = gesture_dict[0]
            elif thumb_tip[1] < index_tip[1] and thumb_tip[0] < index_tip[0]:
                gesture_name = gesture_dict[3]

                # Display the recognized gesture name
            cv2.putText(frame, gesture_name, (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)
    # Check for user input to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()


