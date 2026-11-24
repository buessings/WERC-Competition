import cv2
import numpy as np
from pupil_apriltags import Detector

# Use index 1 for external Logitech USB webcam on HP laptop
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Lock 720p resolution and disable autofocus
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 15)  # 0 = far/infinity, 15-30 = arm's length testing

# Adjusted intrinsics based on your 10 cm vs 25 cm test (920.0 * 0.10 / 0.25 = 368.0)
# NOTE: If your black square is actually 4 cm (0.04 m) instead of 10 cm (0.10 m),
# change camera_params back to (920.0, 920.0, 640.0, 360.0) and set TAG_SIZE_METERS = 0.04
camera_params = (368.0, 368.0, 640.0, 360.0)

# Exact width of your tag's black square in meters (0.10 = 10 cm, 0.04 = 4 cm)
TAG_SIZE_METERS = 0.10

detector = Detector(families="tag36h11")

print("Starting AprilTag sensor on Logitech webcam... Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera 1. Check USB connection or try index 0.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_METERS,
    )

    for tag in tags:
        # Draw green outline around the tag
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

        # Extract X (left/right), Y (up/down), Z (distance forward) in meters
        x, y, z = tag.pose_t.ravel()
        label = f"ID {tag.tag_id}: Z={z:.2f}m ({z*100:.1f}cm) | X={x:.2f}m, Y={y:.2f}m"
        cv2.putText(
            frame,
            label,
            (int(corners[0][0]), int(corners[0][1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow("HP Laptop - Logitech AprilTag Sensor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()