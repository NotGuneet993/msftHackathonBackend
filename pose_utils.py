import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_pose_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    results_list = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if results.pose_landmarks:
                landmarks = [
                    (lm.x, lm.y, lm.z, lm.visibility)
                    for lm in results.pose_landmarks.landmark
                ]
                results_list.append(landmarks)
            else:
                results_list.append(None)
    cap.release()
    return results_list
