import os
import cv2
import mediapipe as mp
import pandas as pd

# BEFORE RUNNING 

# you need an input dir called "rawVideos"
# you need an output dir called "outputs"

# make sure all your name_desc_#.mp4 
desc = "halfsquat"       # example: mine is "buttwink"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
    else:
        return [(None, None, None, None)] * 33

def process_video_to_csv(video_path, csv_name):
    cap = cv2.VideoCapture(video_path)
    data = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extract_landmarks_from_frame(frame)
        flat = [coord for lm in landmarks for coord in lm]
        data.append({"frame": frame_num, "landmarks": flat})
        frame_num += 1

    cap.release()

    # Prepare CSV
    rows = []
    for entry in data:
        row = {"frame": entry["frame"]}
        for i in range(33):
            x, y, z, vis = entry["landmarks"][i * 4:(i + 1) * 4]
            row[f"lm_{i}_x"] = x
            row[f"lm_{i}_y"] = y
            row[f"lm_{i}_z"] = z
            row[f"lm_{i}_vis"] = vis
        rows.append(row)

    df = pd.DataFrame(rows)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

# Only run this if the script is executed directly, not on import
if __name__ == "__main__":
    raw_videos_dir = "rawVideos"
    if not os.path.exists(raw_videos_dir):
        print(f"Directory '{raw_videos_dir}' does not exist. Exiting.")
    else:
        counter = 1
        for filename in sorted(os.listdir(raw_videos_dir)):
            if not filename.endswith(".mp4"):
                continue
            base = filename.lower().replace(".mp4", "")
            parts = base.split("_")
            if len(parts) == 3 and parts[1] == desc:
                output_csv = f"{desc}_{counter}.csv"
                counter += 1
                video_path = os.path.join(raw_videos_dir, filename)
                process_video_to_csv(video_path, output_csv)

