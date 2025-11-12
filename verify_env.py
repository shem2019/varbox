
import cv2, mediapipe as mp
from fpdf import FPDF
import numpy as np, tempfile, os

print("[OK] cv2", cv2.__version__)
print("[OK] mediapipe", mp.__version__)
print("[OK] fpdf", FPDF.__version__)

# Write tiny video to confirm codec path works
h,w=120,160; tmp=os.path.join(tempfile.gettempdir(),"varbox_probe.mp4")
fcc=cv2.VideoWriter_fourcc(*"mp4v"); vw=cv2.VideoWriter(tmp,fcc,24,(w,h))
assert vw.isOpened(), "VideoWriter failed"
for _ in range(10): vw.write(np.zeros((h,w,3),np.uint8))
vw.release()
print("[OK] wrote:", tmp)


