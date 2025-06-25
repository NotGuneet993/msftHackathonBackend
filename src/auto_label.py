import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from glob import glob

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

landmark_names = [lm.name.lower() for lm in mp_pose.PoseLandmark]
issue_labels = ['leaning_forward', 'knees_in_left', 'knees_in_right', 'knees_in_both', 'butt_wink', 'half_squat']

def extract_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    landmarks = []

    if result.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        landmarks = [(None, None, None, None)] * 33

    return landmarks, frame

def process_video(video_path, duration_secs=2.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames = int(fps * duration_secs)
    available_frames = min(total_frames, num_frames)
    frame_indices = np.linspace(0, available_frames - 1, num=available_frames, dtype=int)

    data = []
    output_frames = []
    frame_num = 0
    last_valid_frame = None
    last_landmarks = None

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            if last_valid_frame is not None and last_landmarks is not None:
                data.append({"frame": frame_num, "landmarks": last_landmarks})
                output_frames.append(last_valid_frame.copy())
                frame_num += 1
            continue

        landmarks, annotated = extract_landmarks_from_frame(frame)
        flat = [coord for lm in landmarks for coord in lm]
        data.append({"frame": frame_num, "landmarks": flat})
        output_frames.append(annotated)
        last_valid_frame = annotated
        last_landmarks = flat
        frame_num += 1

    while len(data) < num_frames and last_valid_frame is not None and last_landmarks is not None:
        data.append({"frame": frame_num, "landmarks": last_landmarks})
        output_frames.append(last_valid_frame.copy())
        frame_num += 1

    cap.release()
    return data, output_frames

def infer_label_and_issue(filename):
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) < 2:
        return 'good', 'none'

    issue_part = parts[1].lower()
    mapping = {
        'kneesinleft': 'knees_in_left',
        'kneesinright': 'knees_in_right',
        'kneesinboth': 'knees_in_both',
        'leaningforward': 'leaning_forward',
        'buttwink': 'butt_wink',
        'halfsquat': 'half_squat'
    }

    if issue_part in mapping:
        return 'bad', mapping[issue_part]
    else:
        return 'good', 'none'

def save_to_csv(data, output_path, label, issue, video_id):
    rows = []
    for entry in data:
        row = {
            "frame": entry["frame"],
            "video": video_id,
            "label": label,
            "issue": issue
        }
        for i, name in enumerate(landmark_names):
            x, y, z, vis = entry["landmarks"][i * 4:(i + 1) * 4]
            row[f"{name}_x"] = x
            row[f"{name}_y"] = y
            row[f"{name}_z"] = z
            row[f"{name}_vis"] = vis
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to {output_path}")

def save_video(frames, video_name, output_dir="../videos/annotated"):
    os.makedirs(output_dir, exist_ok=True)
    height, width, _ = frames[0].shape
    out_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 15, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved annotated video to {out_path}")

def main():
    input_dir = "../videos/raw"
    output_csv_dir = "../data/csvs"
    os.makedirs(output_csv_dir, exist_ok=True)

    videos = sorted(glob(os.path.join(input_dir, "*.mp4")))

    for video_path in videos:
        base = os.path.splitext(os.path.basename(video_path))[0].lower()
        print(f"\n=== Processing {base}: {os.path.basename(video_path)} ===")

        label, issue = infer_label_and_issue(video_path)
        data, annotated_frames = process_video(video_path)

        save_to_csv(data, os.path.join(output_csv_dir, f"{base}.csv"), label, issue, base)
        save_video(annotated_frames, base)

        try:
            os.remove(video_path)
            print(f"Deleted {video_path}")
        except Exception as e:
            print(f"Could not delete {video_path}: {e}")

if __name__ == "__main__":
    main()
