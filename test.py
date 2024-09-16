from ultralytics import YOLO
import cv2

model= YOLO('best.pt')
model.predict(source='/Users/adit/Desktop/Detection/test_videos/night_drive.mp4',show=True,conf=0.2)