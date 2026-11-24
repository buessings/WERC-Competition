import cv2
import numpy as np
from pupil_apriltags import Detector

# ==========================================
# 1. ROVER MOTOR INTERFACE (CUSTOMIZE THIS)
# ==========================================
def send_rover_commands(forward_speed, turn_rate):
    """
    forward_speed: -1.0 (full reverse) to +1.0 (full forward)
    turn_rate:     -1.0 (full left) to +1.0 (full right)
    """
    left_motor = np.clip(forward_speed + turn_rate, -1.0, 1.0)
    right_motor = np.clip(forward_speed - turn_rate, -1.0, 1.0)

    # TODO: Replace with your serial/UDP/GPIO motor command
    print(f"CMD -> Forward: {forward_speed:+.2f} | Turn: {turn_rate:+.2f} | L:{left_motor:+.2f} R:{right_motor:+.2f}")

# ==========================================
# 2. DOCKING PARAMETERS
# ==========================================
TARGET_STOP_CM = 5.0        # Stop 5 cm in front of the tag
SLOW_DOWN_OFFSET_CM = 20.0  # Start slowing down 20 cm before stop target (at 25 cm)
SLOW_DOWN_START_CM = TARGET_STOP_CM + SLOW_DOWN_OFFSET_CM

TAG_PITCH_DEG = 10.0        # Fixed 10-degree forward/backward dust-shedding tilt

MAX_SPEED = 0.60            # Maximum forward speed (0.0 to 1.0)
MIN_CREEP_SPEED = 0.15      # Minimum speed so motors don't stall near the end

KP_LATERAL = 0.015          # Steering gain for horizontal offset (X in cm)
KP_ANGLE = 0.012            # Steering gain to keep left-right side angle at 0 deg
ALIGN_TOLERANCE_CM = 1.5    # Considered "straight in front" within +/- 1.5 cm
SIDE_ANGLE_TOL_DEG = 3.0    # Allowed left-right angle error (+/- 3 deg)

# ==========================================
# 3. CAMERA & GEOMETRY HELPERS
# ==========================================
def get_tag_angles(R):
    """
    Uses the tag's Z-axis normal vector (3rd column of R) to cleanly separate
    left-right side angle from vertical dust-shedding tilt without gimbal cross-talk.
    """
    nx, ny, nz = R[:, 2]
    # Left-right angle on the ground plane (should be 0 deg when rover is straight ahead)
    side_angle_deg = np.degrees(np.arctan2(nx, abs(nz)))
    # Up-down vertical tilt angle (should read ~10 deg)
    vertical_tilt_deg = np.degrees(np.arctan2(ny, abs(nz)))
    return side_angle_deg, vertical_tilt_deg

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 15)

camera_params = (385.1, 385.1, 640.0, 360.0)
TAG_SIZE_METERS = 0.10
detector = Detector(families="tag36h11")

print("Rover AprilTag Docking Active (10-deg Pitched Tag)... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            send_rover_commands(0.0, 0.0)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=TAG_SIZE_METERS,
        )

        if len(tags) == 0:
            send_rover_commands(0.0, 0.0)
            cv2.putText(frame, "NO TAG - STOPPED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Pick the closest detected tag
            tag = min(tags, key=lambda t: t.pose_t[2])

            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

            x_cm = tag.pose_t[0] * 100.0
            slant_z_cm = tag.pose_t[2] * 100.0

            # Project slant distance onto flat floor distance using the 10-deg tilt
            ground_z_cm = slant_z_cm * np.cos(np.radians(TAG_PITCH_DEG))

            # Extract separated side angle (target = 0 deg) and vertical tilt (~10 deg)
            side_angle_deg, vertical_tilt_deg = get_tag_angles(tag.pose_R)

            # --- STEERING CALCULATION (IGNORES 10-DEG PITCH, ENFORCES 0-DEG SIDE ANGLE) ---
            lateral_error = x_cm
            side_angle_error = side_angle_deg if ground_z_cm > 15.0 else 0.0
            turn_rate = np.clip((KP_LATERAL * lateral_error) - (KP_ANGLE * side_angle_error), -0.5, 0.5)

            # --- FORWARD SPEED & BRAKING CALCULATION ---
            dist_remaining = ground_z_cm - TARGET_STOP_CM

            if dist_remaining <= 0:
                forward_speed = 0.0
                if abs(lateral_error) <= ALIGN_TOLERANCE_CM and abs(side_angle_deg) <= SIDE_ANGLE_TOL_DEG:
                    turn_rate = 0.0
                    status = "DOCKED AT 5 CM (STRAIGHT)"
                else:
                    status = "FINAL SIDE ALIGNMENT"
            elif ground_z_cm <= SLOW_DOWN_START_CM:
                ratio = dist_remaining / SLOW_DOWN_OFFSET_CM
                forward_speed = MIN_CREEP_SPEED + ratio * (MAX_SPEED - MIN_CREEP_SPEED)
                status = "SLOWING DOWN"
            else:
                turn_penalty = max(0.4, 1.0 - abs(turn_rate))
                forward_speed = MAX_SPEED * turn_penalty
                status = "APPROACHING"

            send_rover_commands(forward_speed, turn_rate)

            # Draw live telemetry on screen
            line1 = f"{status} | Floor Dist: {ground_z_cm:.1f}cm | Offset X: {x_cm:.1f}cm"
            line2 = f"Side Angle: {side_angle_deg:.1f}deg (Target 0) | Pitch Tilt: {vertical_tilt_deg:.1f}deg"
            cv2.putText(frame, line1, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, line2, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("Rover AprilTag Docking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    send_rover_commands(0.0, 0.0)
    cap.release()
    cv2.destroyAllWindows()
