import cv2
import mediapipe as mp

# Initialize MediaPipe pose and drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Boost performance

        # Run MediaPipe Pose on the frame
        results = pose.process(image)

        # Draw results back on original BGR image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Draw skeleton on image
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS)

            # Print coordinates of the left knee (as example)
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            h, w, _ = image.shape
            x_px = int(left_knee.x * w)
            y_px = int(left_knee.y * h)
            print(f"Left knee coordinates: ({x_px}, {y_px})")

        # Show the frame with landmarks
        cv2.imshow('MediaPipe Pose', image)

        # Exit if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
