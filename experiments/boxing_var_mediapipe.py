import cv2
import mediapipe as mp
import numpy as np
import fpdf

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- Video Setup ---
video_path = "assets/boxing_match.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"❌ Video file not found: {video_path}")

frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("boxing_output_mediapipe.mp4", fourcc, frame_rate, (frame_width, frame_height))

# --- Scoring ---
PUNCH_DISTANCE_THRESHOLD = 50
fighter1_score, fighter2_score = 0, 0
punch_log = []
frame_count = 0

# Cooldown trackers to prevent duplicate scoring per punch
cooldown_frames = 15
f1_last_punch = 0
f2_last_punch = 0

# --- Helper Functions ---
def get_landmark_xy(landmarks, idx):
    return [int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)]

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- Process Video Frame-by-Frame ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Define wrists and head
        left_wrist = get_landmark_xy(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_landmark_xy(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        head = get_landmark_xy(landmarks, mp_pose.PoseLandmark.NOSE)

        # Fake second fighter (mirror)
        # You can replace this with a second detection or real opponent logic
        mirror_left_wrist = [frame_width - right_wrist[0], right_wrist[1]]
        mirror_right_wrist = [frame_width - left_wrist[0], left_wrist[1]]
        mirror_head = [frame_width - head[0], head[1]]

        # Fighter 1 Punch Detection
        if frame_count - f1_last_punch > cooldown_frames:
            if calculate_distance(left_wrist, mirror_head) < PUNCH_DISTANCE_THRESHOLD \
               or calculate_distance(right_wrist, mirror_head) < PUNCH_DISTANCE_THRESHOLD:
                fighter1_score += 1
                time_stamp = f"{frame_count // frame_rate:02}:{frame_count % frame_rate:02}"
                punch_log.append(("Fighter 1", time_stamp, fighter1_score))
                f1_last_punch = frame_count

        # Fighter 2 Punch Detection (mirror)
        if frame_count - f2_last_punch > cooldown_frames:
            if calculate_distance(mirror_left_wrist, head) < PUNCH_DISTANCE_THRESHOLD \
               or calculate_distance(mirror_right_wrist, head) < PUNCH_DISTANCE_THRESHOLD:
                fighter2_score += 1
                time_stamp = f"{frame_count // frame_rate:02}:{frame_count % frame_rate:02}"
                punch_log.append(("Fighter 2", time_stamp, fighter2_score))
                f2_last_punch = frame_count

        # Draw overlay
        cv2.putText(frame, f"F1 Score: {fighter1_score}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"F2 Score: {fighter2_score}", (frame_width - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        for idx in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.NOSE]:
            cx, cy = get_landmark_xy(landmarks, idx)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

    out.write(frame)

cap.release()
out.release()
pose.close()
print("✅ Video processing complete and saved as 'boxing_output_mediapipe.mp4'")

# --- Generate PDF Scorecard ---
pdf = fpdf.FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Boxing Match Scorecard", ln=True, align="C")

pdf.set_font("Arial", size=12)
pdf.cell(50, 10, "Fighter", 1)
pdf.cell(50, 10, "Time (MM:SS)", 1)
pdf.cell(50, 10, "Punch Count", 1)
pdf.ln()

for fighter, time, punch_count in punch_log:
    pdf.cell(50, 10, fighter, 1)
    pdf.cell(50, 10, time, 1)
    pdf.cell(50, 10, str(punch_count), 1)
    pdf.ln()

pdf.output("boxing_scorecard.pdf")
print("✅ Scorecard saved as 'boxing_scorecard.pdf'")
