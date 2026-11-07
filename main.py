import cv2
import pygame
import json
import os

from ultralytics import YOLO
from collections import deque
from datetime import datetime


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PPE_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
FIRE_MODEL_PATH = os.path.join(MODEL_DIR, "fire_smoke_best.pt")

PPE_ALARM_PATH = os.path.join(BASE_DIR, "ppe_alarm.wav")
FIRE_ALARM_PATH = os.path.join(BASE_DIR, "fire_alarm.wav")

EVENT_FILE = os.path.join(BASE_DIR, "events.json")


# =========================================================
# LOAD MODELS
# =========================================================

if not os.path.exists(PPE_MODEL_PATH):
    raise FileNotFoundError(f"PPE model not found: {PPE_MODEL_PATH}")

if not os.path.exists(FIRE_MODEL_PATH):
    raise FileNotFoundError(f"Fire/smoke model not found: {FIRE_MODEL_PATH}")

ppe_model = YOLO(PPE_MODEL_PATH)
fire_model = YOLO(FIRE_MODEL_PATH)

print("Both models loaded successfully")
print("PPE model:", ppe_model.names)
print("Fire model:", fire_model.names)


# =========================================================
# ALARMS
# =========================================================

pygame.mixer.init()

ppe_alarm = pygame.mixer.Sound(PPE_ALARM_PATH)
fire_alarm = pygame.mixer.Sound(FIRE_ALARM_PATH)


# =========================================================
# PPE SETTINGS
# =========================================================

required_ppe = [
    "helmet",
    "safety-vest",
    "gloves",
    "shoes"
]

PPE_CONFIDENCE = 0.40
PPE_VOTE_WINDOW = 5
PPE_VOTE_THRESHOLD = 3


# =========================================================
# FIRE / SMOKE SETTINGS
# =========================================================

FIRE_CONFIDENCE = 0.40
FIRE_VOTE_WINDOW = 5
FIRE_VOTE_THRESHOLD = 3

FIRE_RESET_FRAMES = 10
SMOKE_RESET_FRAMES = 10


# =========================================================
# EVENT SETTINGS
# =========================================================

CAMERA_ID = "CAM_01"


# =========================================================
# SAVE EVENT
# =========================================================

def save_event(event):

    events = []

    if os.path.exists(EVENT_FILE):

        try:
            with open(EVENT_FILE, "r", encoding="utf-8") as file:
                events = json.load(file)

            if not isinstance(events, list):
                events = []

        except (json.JSONDecodeError, OSError):
            events = []

    events.append(event)

    temp_file = EVENT_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(events, file, indent=4)

    os.replace(temp_file, EVENT_FILE)


# =========================================================
# MAIN VIDEO PROCESSING
# =========================================================

def process_video(video_path):

    # Initialize events
    events = []

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return events

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return events

    print(f"Processing video: {video_path}")

    # Event states
    ppe_violation_active = False
    fire_event_active = False
    smoke_event_active = False

    # Temporal voting histories
    ppe_history = {
        item: deque(maxlen=PPE_VOTE_WINDOW)
        for item in required_ppe
    }

    fire_history = deque(maxlen=FIRE_VOTE_WINDOW)
    smoke_history = deque(maxlen=FIRE_VOTE_WINDOW)

    fire_safe_frames = 0
    smoke_safe_frames = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Video ended or frame could not be read.")
                break

            original_frame = frame.copy()

            # =================================================
            # PPE DETECTION
            # =================================================

            ppe_results = ppe_model.predict(
                frame,
                verbose=False,
                conf=PPE_CONFIDENCE
            )

            current_ppe = set()
            person_detected = False

            for result in ppe_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = ppe_model.names[cls_id]

                    if confidence < PPE_CONFIDENCE:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    if class_name == "person":
                        person_detected = True

                    elif class_name in required_ppe:
                        current_ppe.add(class_name)

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{class_name} {confidence:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # =================================================
            # PPE TEMPORAL VOTING
            # =================================================

            confirmed_missing_ppe = []

            if person_detected:

                for item in required_ppe:

                    ppe_history[item].append(
                        1 if item in current_ppe else 0
                    )

                for item in required_ppe:

                    history = ppe_history[item]

                    if (
                        len(history) >= PPE_VOTE_THRESHOLD
                        and history.count(0) >= PPE_VOTE_THRESHOLD
                    ):

                        confirmed_missing_ppe.append(item)

            else:

                for history in ppe_history.values():
                    history.clear()

            # =================================================
            # PPE EVENT
            # =================================================

            if confirmed_missing_ppe:

                if not ppe_violation_active:

                    ppe_event = {

                        "event_type": "PPE_VIOLATION",

                        "camera_id": CAMERA_ID,

                        "status": "NON-COMPLIANT",

                        "missing_ppe": confirmed_missing_ppe.copy(),

                        "severity": "HIGH",

                        "timestamp": datetime.now().isoformat()

                    }

                    print(ppe_event)

                    save_event(ppe_event)

                    events.append(ppe_event)

                    ppe_alarm.play()

                    ppe_violation_active = True

            else:

                ppe_violation_active = False

            # =================================================
            # FIRE / SMOKE DETECTION
            # =================================================

            fire_results = fire_model.predict(
                original_frame,
                verbose=False,
                conf=FIRE_CONFIDENCE
            )

            fire_detected_this_frame = False
            smoke_detected_this_frame = False

            for result in fire_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = fire_model.names[cls_id]

                    if confidence < FIRE_CONFIDENCE:
                        continue

                    if class_name == "fire":
                        fire_detected_this_frame = True

                    elif class_name == "smoke":
                        smoke_detected_this_frame = True

                    # Draw fire/smoke box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{class_name} {confidence:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            # =================================================
            # FIRE / SMOKE TEMPORAL VOTING
            # =================================================

            fire_history.append(
                1 if fire_detected_this_frame else 0
            )

            smoke_history.append(
                1 if smoke_detected_this_frame else 0
            )

            fire_votes = sum(fire_history)
            smoke_votes = sum(smoke_history)

            confirmed_fire = (
                len(fire_history) >= FIRE_VOTE_THRESHOLD
                and fire_votes >= FIRE_VOTE_THRESHOLD
            )

            confirmed_smoke = (
                len(smoke_history) >= FIRE_VOTE_THRESHOLD
                and smoke_votes >= FIRE_VOTE_THRESHOLD
            )

            # =================================================
            # FIRE EVENT
            # =================================================

            if confirmed_fire:

                fire_safe_frames = 0

                if not fire_event_active:

                    fire_event = {

                        "event_type": "FIRE_DETECTED",

                        "camera_id": CAMERA_ID,

                        "status": "DANGER",

                        "severity": "CRITICAL",

                        "timestamp": datetime.now().isoformat()

                    }

                    print(fire_event)

                    save_event(fire_event)

                    events.append(fire_event)

                    fire_alarm.play()

                    fire_event_active = True

            elif fire_event_active:

                fire_safe_frames += 1

                if fire_safe_frames >= FIRE_RESET_FRAMES:

                    fire_event_active = False

                    fire_safe_frames = 0

            # =================================================
            # SMOKE EVENT
            # =================================================

            if confirmed_smoke:

                smoke_safe_frames = 0

                if not smoke_event_active:

                    smoke_event = {

                        "event_type": "SMOKE_DETECTED",

                        "camera_id": CAMERA_ID,

                        "status": "DANGER",

                        "severity": "HIGH",

                        "timestamp": datetime.now().isoformat()

                    }

                    print(smoke_event)

                    save_event(smoke_event)

                    events.append(smoke_event)

                    fire_alarm.play()

                    smoke_event_active = True

            elif smoke_event_active:

                smoke_safe_frames += 1

                if smoke_safe_frames >= SMOKE_RESET_FRAMES:

                    smoke_event_active = False

                    smoke_safe_frames = 0

            # =================================================
            # DISPLAY TITLE
            # =================================================

            cv2.putText(
                frame,
                "AI SAFETY MONITORING",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # =================================================
            # DISPLAY PPE STATUS
            # =================================================

            if confirmed_missing_ppe:

                ppe_status = (
                    "PPE: CHECK REQUIRED | Missing: "
                    + ", ".join(confirmed_missing_ppe)
                )

                ppe_color = (0, 0, 255)

            elif person_detected:

                ppe_status = "PPE: NO MISSING ITEMS DETECTED"

                ppe_color = (0, 255, 0)

            else:

                ppe_status = "PPE: NO PERSON DETECTED"

                ppe_color = (255, 255, 255)

            cv2.putText(
                frame,
                ppe_status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                ppe_color,
                2
            )

            # =================================================
            # DISPLAY FIRE / SMOKE STATUS
            # =================================================

            if confirmed_fire:

                fire_status = (
                    f"FIRE DETECTED ({fire_votes}/{FIRE_VOTE_WINDOW})"
                )

                fire_color = (0, 0, 255)

            elif confirmed_smoke:

                fire_status = (
                    f"SMOKE DETECTED ({smoke_votes}/{FIRE_VOTE_WINDOW})"
                )

                fire_color = (0, 0, 255)

            else:

                fire_status = "FIRE/SMOKE: NO DETECTION CONFIRMED"

                fire_color = (0, 255, 0)

            cv2.putText(
                frame,
                fire_status,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                fire_color,
                2
            )

            # =================================================
            # DISPLAY FRAME
            # =================================================

            display_frame = frame.copy()

            height, width = display_frame.shape[:2]

            max_width = 1280
            max_height = 720

            scale = min(
                max_width / width,
                max_height / height,
                1.0
            )

            new_width = int(width * scale)
            new_height = int(height * scale)

            display_frame = cv2.resize(
                display_frame,
                (new_width, new_height)
            )

            cv2.imshow(
                "BPUT AI Safety Monitoring",
                display_frame
            )

            # =================================================
            # PRESS Q TO EXIT
            # =================================================

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        pygame.mixer.quit()

        print("System stopped")

    return events


# =========================================================
# RUN PROJECT
# =========================================================

if __name__ == "__main__":

    video_path = os.path.join(
        INPUT_DIR,
        "construction.mp4"
    )

    process_video(video_path)