import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
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

        cv2.imshow('Pose Preview', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    while len(data) < num_frames and last_valid_frame is not None and last_landmarks is not None:
        data.append({"frame": frame_num, "landmarks": last_landmarks})
        output_frames.append(last_valid_frame.copy())
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    return data, output_frames

def label_video_cli():
    print("\n=== Squat Labeling ===")
    label = input("Overall form? (good / bad): ").strip().lower()

    issue = "none"
    if label == "bad":
        print("Choose issue type:")
        for i, lbl in enumerate(issue_labels):
            print(f"{i+1}. {lbl}")
        selected = int(input("Enter number: "))
        issue = issue_labels[selected - 1]

    return label, issue

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

def get_next_index(csv_dir):
    existing = sorted(glob(os.path.join(csv_dir, "squat*.csv")))
    if not existing:
        return 1
    last_file = os.path.basename(existing[-1])
    last_num = int(''.join(filter(str.isdigit, last_file)))
    return last_num + 1

def main():
    input_dir = "../videos/raw"
    output_csv_dir = "../data/csvs"
    os.makedirs(output_csv_dir, exist_ok=True)

    videos = sorted(glob(os.path.join(input_dir, "*.mp4")))
    start_index = get_next_index(output_csv_dir)

    for i, video_path in enumerate(videos):
        squat_num = start_index + i
        video_id = f"squat{squat_num}"
        print(f"\n=== Processing {video_id}: {os.path.basename(video_path)} ===")

        data, annotated_frames = process_video(video_path)
        label, issue = label_video_cli()

        save_to_csv(data, os.path.join(output_csv_dir, f"{video_id}.csv"), label, issue, video_id)
        save_video(annotated_frames, video_id)

        # Delete raw video after successful processing
        try:
            os.remove(video_path)
            print(f"Deleted {video_path}")
        except Exception as e:
            print(f"Could not delete {video_path}: {e}")

if __name__ == "__main__":
    main()
