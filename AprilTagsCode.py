import cv2
import numpy as np
from pupil_apriltags import Detector

def rotation_matrix_to_euler_angles(R):
    # Convert 3x3 rotation matrix into Roll, Pitch, Yaw (in degrees)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# Use index 1 for external Logitech USB webcam on HP laptop
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Lock 720p resolution and disable autofocus
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 15)

# Fine-tuned intrinsics calibrated across 50 cm, 70 cm, and 100 cm
camera_params = (385.1, 385.1, 640.0, 360.0)
TAG_SIZE_METERS = 0.10

detector = Detector(families="tag36h11")

print("Starting AprilTag sensor with tilt tracking... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera 1. Check USB connection.")
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

        # 1. Extract Position (X, Y, Z in meters)
        x, y, z = tag.pose_t.ravel()
        pos_label = f"ID {tag.tag_id}: Z={z*100:.1f}cm | X={x*100:.1f}cm, Y={y*100:.1f}cm"

        # 2. Extract Tilt Angles (Roll, Pitch, Yaw in degrees)
        roll, pitch, yaw = rotation_matrix_to_euler_angles(tag.pose_R)
        tilt_label = f"Tilt -> Roll: {roll:.1f} deg | Pitch: {pitch:.1f} deg | Yaw: {yaw:.1f} deg"

        # Draw text labels above the tag
        top_x, top_y = int(corners[0][0]), int(corners[0][1])
        cv2.putText(frame, pos_label, (top_x, top_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, tilt_label, (top_x, top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("HP Laptop - Logitech AprilTag Sensor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()