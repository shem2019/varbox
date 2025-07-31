# video_processor.py

import cv2
import mediapipe as mp
from config import INPUT_VIDEO, OUTPUT_VIDEO, FRAME_RATE
from punch_detector import detect_punch
from multi_person_tracker import MultiPersonPoseTracker
from score_tracker import ScoreTracker

POSE = mp.solutions.pose.PoseLandmark


def process_video(score_tracker):
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open video: {INPUT_VIDEO}")

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or FRAME_RATE
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, frame_rate, (frame_width, frame_height))

    frame_count = 0
    pose_tracker = MultiPersonPoseTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        poses = pose_tracker.process_frame(frame, frame_count)

        # Match every fighter with all other fighters
        for id1, data1 in poses.items():
            for id2, data2 in poses.items():
                if id1 == id2:
                    continue  # skip self

                kpts1 = data1["keypoints"]
                kpts2 = data2["keypoints"]

                if all(k in kpts1 and k in kpts2 for k in [POSE.LEFT_WRIST, POSE.RIGHT_WRIST, POSE.NOSE]):
                    fighter1 = {
                        "left_wrist": kpts1[POSE.LEFT_WRIST],
                        "right_wrist": kpts1[POSE.RIGHT_WRIST],
                        "head": kpts1[POSE.NOSE]
                    }
                    fighter2_head = kpts2[POSE.NOSE]

                    timestamp = f"{(frame_count // frame_rate):02}:{(frame_count % frame_rate):02}"

                    if detect_punch(fighter1, fighter2_head):
                        score_tracker.update(frame_count, fighter_id=id1, timestamp=timestamp)

        # Draw visual debug overlays
        for boxer_id, data in poses.items():
            kpts = data["keypoints"]
            x1, y1, x2, y2 = data["box"]
            score = score_tracker.get_score(boxer_id)
            nose = kpts.get(POSE.NOSE, (x1, y1))

            # 🟣 Center marker
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)

            # 🟩 Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 🟡 Fighter ID and Score
            cv2.putText(frame, f"ID {boxer_id} | Score: {score}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 🔵 Pose landmarks
            for pt in kpts.values():
                cv2.circle(frame, tuple(pt), 4, (255, 255, 0), -1)

            # 🔷 Optional: Segmentation mask overlay
            mask = data.get("mask")
            if mask is not None:
                try:
                    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                    mask_binary = (mask_resized > 0.1).astype('uint8') * 255
                    mask_rgb = cv2.merge([mask_binary] * 3)
                    roi = frame[y1:y2, x1:x2]
                    overlay = cv2.addWeighted(roi, 0.6, mask_rgb, 0.4, 0)
                    frame[y1:y2, x1:x2] = overlay
                except Exception as e:
                    print(f"⚠️ Mask overlay skipped for ID {boxer_id}: {e}")

            # 📢 Print debug info
            print(f"📍 Boxer {boxer_id} | Box=({x1},{y1},{x2},{y2}) | Score={score} | Keypoints={len(kpts)}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Multi-person video with overlays saved to {OUTPUT_VIDEO}")
