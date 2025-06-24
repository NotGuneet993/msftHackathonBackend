import cv2
import mediapipe as mp
import os
import glob

# Initialize MediaPipe pose and drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define the videos folder path
videos_folder = "videos"

# Supported video formats
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']

def process_video(video_path):
    """Process a single video file"""
    print(f"Processing video: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return True
    
    # Get video properties for better playback
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = int(1000 / fps) if fps > 0 else 33  # Default to ~30fps if fps is 0
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create named window with resizable option
    window_name = f'MediaPipe Pose - {os.path.basename(video_path)}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))
    
    # Custom drawing specifications for better visibility
    landmark_drawing_spec = mp_drawing.DrawingSpec(
        color=(0, 255, 0),  # Green landmarks
        thickness=2,
        circle_radius=2
    )
    connection_drawing_spec = mp_drawing.DrawingSpec(
        color=(0, 255, 255),  # Yellow connections
        thickness=2
    )
    
    with mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=1  # Better accuracy
    ) as pose:
        frame_count = 0
        paused = False
        
        while cap.isOpened():
            if not paused:
                success, frame = cap.read()
                if not success:
                    print(f"Finished processing video: {video_path}")
                    break

                frame_count += 1
                
                # Convert BGR to RGB for MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  # Boost performance

                # Run MediaPipe Pose on the frame
                results = pose.process(image)

                # Draw results back on original BGR image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    # Draw skeleton with custom styling
                    mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=landmark_drawing_spec,
                        connection_drawing_spec=connection_drawing_spec
                    )

                    # Optional: Print coordinates of specific landmarks
                    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                    h, w, _ = image.shape
                    x_px = int(left_knee.x * w)
                    y_px = int(left_knee.y * h)
                    
                    # Add frame counter and coordinates as overlay text
                    cv2.putText(image, f"Frame: {frame_count}/{total_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Left Knee: ({x_px}, {y_px})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Show message when no pose is detected
                    cv2.putText(image, f"Frame: {frame_count}/{total_frames} - No pose detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add control instructions
                cv2.putText(image, "Controls: SPACE=pause, Q=quit, N=next video, R=restart", 
                           (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the frame with landmarks
            cv2.imshow(window_name, image)

            # Handle key presses with proper timing for video playback
            key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
            
            if key == ord('q'):
                cap.release()
                cv2.destroyWindow(window_name)
                return False  # Signal to stop processing all videos
            elif key == ord('n'):
                break  # Move to next video
            elif key == ord(' '):  # Spacebar to pause/unpause
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):  # R to restart video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                paused = False
                print("Restarting video...")

    cap.release()
    cv2.destroyWindow(window_name)
    return True  # Continue processing

def main():
    # Check if videos folder exists
    if not os.path.exists(videos_folder):
        print(f"Videos folder '{videos_folder}' not found. Creating it...")
        os.makedirs(videos_folder)
        print(f"Please place your video files in the '{videos_folder}' folder and run the script again.")
        return

    # Get all video files from the videos folder
    video_files = []
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_folder, extension)))
    
    if not video_files:
        print(f"No video files found in '{videos_folder}' folder.")
        print(f"Supported formats: {', '.join([ext[2:] for ext in video_extensions])}")
        return

    print(f"Found {len(video_files)} video file(s):")
    for video in video_files:
        print(f"  - {video}")
    
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'n' to skip to next video")
    print("  - Press SPACE to pause/unpause")
    print("  - Press 'r' to restart current video")
    print()

    # Process each video file
    for video_path in video_files:
        if not process_video(video_path):
            break  # User pressed 'q' to quit
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()