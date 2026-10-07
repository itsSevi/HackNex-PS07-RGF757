# ============================================================
# HACKNEX 2026 - PS07
# AUTONOMOUS VISION & BEHAVIOUR UNDERSTANDING
#
# YOLO POSE + BoT-SORT + ReID
# + STABLE IDENTITY MANAGER
#
# Purpose:
# Keep the same Person ID when another person temporarily
# passes in front of them.
# ============================================================

import cv2
import os
import csv
import math
from collections import defaultdict, Counter
from datetime import datetime

from ultralytics import YOLO


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_FOLDER = os.path.dirname(
    SCRIPT_DIR
)

VIDEO_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "test_video"
)

MODEL_NAME = "yolo26n-pose.pt"

TRACKER_CONFIG = os.path.join(
    SCRIPT_DIR,
    "botsort_reid.yaml"
)


# ============================================================
# 2. SETTINGS
# ============================================================

DETECTION_CONFIDENCE = 0.30
POSE_CONFIDENCE = 0.20

STATIONARY_THRESHOLD = 5.0

MOVEMENT_DISTANCE_THRESHOLD = 3.0

MOVEMENT_HISTORY_FRAMES = 6
POSTURE_HISTORY_FRAMES = 7

# Process every second frame
FRAME_SKIP = 2


# ============================================================
# 3. STABLE ID SETTINGS
# ============================================================

# How long a person can disappear and still be remembered
IDENTITY_MEMORY_SECONDS = 12.0

# Maximum distance allowed when reconnecting an identity
MAX_IDENTITY_DISTANCE = 220

# Appearance matching importance
APPEARANCE_WEIGHT = 0.50

# Position matching importance
POSITION_WEIGHT = 0.25

# Size matching importance
SIZE_WEIGHT = 0.15

# Direction matching importance
DIRECTION_WEIGHT = 0.10


# ============================================================
# 4. CHECK TRACKER CONFIG
# ============================================================

if not os.path.exists(TRACKER_CONFIG):

    print("\n❌ botsort_reid.yaml NOT FOUND")

    print("\nExpected:")
    print(TRACKER_CONFIG)

    input("\nPress ENTER to close...")

    raise SystemExit


print("\n✅ BoT-SORT configuration found.")


# ============================================================
# 5. FIND VIDEO
# ============================================================

print("\n" + "=" * 70)
print("HACKNEX PS07")
print("STABLE PERSON TRACKING")
print("=" * 70)

print("\nSearching for video...")

print(VIDEO_FOLDER)


if not os.path.exists(VIDEO_FOLDER):

    print("\n❌ Test video folder not found.")

    print(VIDEO_FOLDER)

    input("\nPress ENTER to close...")

    raise SystemExit


video_files = []


for filename in os.listdir(VIDEO_FOLDER):

    full_path = os.path.join(
        VIDEO_FOLDER,
        filename
    )

    if os.path.isfile(full_path):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv"
        ]:

            video_files.append(
                full_path
            )


if not video_files:

    print("\n❌ No video found.")

    input("\nPress ENTER to close...")

    raise SystemExit


# Prefer test.mp4.mp4
VIDEO_PATH = None


for video in video_files:

    if os.path.basename(
        video
    ).lower() == "test.mp4.mp4":

        VIDEO_PATH = video

        break


if VIDEO_PATH is None:

    for video in video_files:

        if os.path.basename(
            video
        ).lower().startswith("test"):

            VIDEO_PATH = video

            break


if VIDEO_PATH is None:

    VIDEO_PATH = video_files[0]


print("\n✅ VIDEO FOUND")

print(VIDEO_PATH)


# ============================================================
# 6. LOAD YOLO
# ============================================================

print("\nLoading YOLO Pose model...")

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    MODEL_NAME
)


if os.path.exists(MODEL_PATH):

    model = YOLO(
        MODEL_PATH
    )

else:

    model = YOLO(
        MODEL_NAME
    )


print("✅ YOLO Pose loaded.")


# ============================================================
# 7. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    print("\n❌ Could not open video.")

    input("\nPress ENTER to close...")

    raise SystemExit


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 30.0


frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("\nVideo:")
print(
    f"{frame_width} x {frame_height}"
)

print(
    f"{fps:.1f} FPS"
)

print(
    f"{total_frames} frames"
)


# ============================================================
# 8. OUTPUT
# ============================================================

OUTPUT_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "ps07_output"
)

EVIDENCE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "evidence"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    EVIDENCE_FOLDER,
    exist_ok=True
)


OUTPUT_VIDEO = os.path.join(
    OUTPUT_FOLDER,
    "ps07_stable_identity.mp4"
)

CSV_FILE = os.path.join(
    OUTPUT_FOLDER,
    "events.csv"
)


# ============================================================
# 9. VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (
        frame_width,
        frame_height
    )
)


# ============================================================
# 10. NORMAL TRACKING MEMORY
# ============================================================

position_history = defaultdict(list)

movement_history = defaultdict(list)

posture_history = defaultdict(list)

stationary_start_time = {}

alert_saved = set()

event_logged = set()


# ============================================================
# 11. STABLE ID MEMORY
# ============================================================

# tracker ID -> stable ID
tracker_to_stable = {}


# stable ID -> memory
stable_people = {}


next_stable_id = 1


# ============================================================
# 12. APPEARANCE
# ============================================================

def get_appearance(
    frame,
    x1,
    y1,
    x2,
    y2
):

    try:

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            return None

        height = crop.shape[0]

        # Use upper 70% of body
        upper = crop[
            0:max(
                1,
                int(height * 0.70)
            ),
            :
        ]

        hsv = cv2.cvtColor(
            upper,
            cv2.COLOR_BGR2HSV
        )

        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [16, 16],
            [0, 180, 0, 256]
        )

        cv2.normalize(
            hist,
            hist
        )

        return hist

    except Exception:

        return None


# ============================================================
# 13. APPEARANCE SIMILARITY
# ============================================================

def appearance_similarity(
    hist1,
    hist2
):

    if hist1 is None or hist2 is None:

        return 0.0

    try:

        value = cv2.compareHist(
            hist1,
            hist2,
            cv2.HISTCMP_CORREL
        )

        value = (
            value + 1.0
        ) / 2.0

        return max(
            0.0,
            min(
                1.0,
                float(value)
            )
        )

    except Exception:

        return 0.0


# ============================================================
# 14. DISTANCE
# ============================================================

def distance(
    p1,
    p2
):

    return math.sqrt(
        (
            p1[0] - p2[0]
        ) ** 2
        +
        (
            p1[1] - p2[1]
        ) ** 2
    )


# ============================================================
# 15. SIZE SIMILARITY
# ============================================================

def size_similarity(
    size1,
    size2
):

    if size1 <= 0 or size2 <= 0:

        return 0.0

    ratio = min(
        size1,
        size2
    ) / max(
        size1,
        size2
    )

    return float(ratio)


# ============================================================
# 16. SMOOTHING
# ============================================================

def smooth_value(
    values
):

    if not values:

        return None

    return Counter(
        values
    ).most_common(
        1
    )[0][0]


# ============================================================
# 17. VIDEO TIME
# ============================================================

def video_time(
    frame_number
):

    seconds = (
        frame_number / fps
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# 18. STABLE ID MANAGER
# ============================================================

def get_stable_id(
    tracker_id,
    center,
    width,
    height,
    appearance,
    current_time
):

    global next_stable_id


    # ========================================================
    # CASE 1
    # Tracker already known
    # ========================================================

    if tracker_id in tracker_to_stable:

        stable_id = tracker_to_stable[
            tracker_id
        ]

        if stable_id in stable_people:

            data = stable_people[
                stable_id
            ]

            # Calculate movement direction
            old_center = data[
                "center"
            ]

            dx = (
                center[0]
                -
                old_center[0]
            )

            dy = (
                center[1]
                -
                old_center[1]
            )

            data["direction"] = (
                dx,
                dy
            )

            data["center"] = center

            data["width"] = width

            data["height"] = height

            data["last_seen"] = (
                current_time
            )

            data["active_tracker"] = (
                tracker_id
            )

            if appearance is not None:

                data[
                    "appearance"
                ] = appearance

        return stable_id


    # ========================================================
    # CASE 2
    # New tracker ID
    #
    # Try to reconnect to an existing person
    # ========================================================

    best_id = None

    best_score = -1


    for stable_id, data in stable_people.items():

        time_missing = (
            current_time
            -
            data["last_seen"]
        )


        # Too long ago
        if (
            time_missing
            >
            IDENTITY_MEMORY_SECONDS
        ):

            continue


        # Current center
        old_center = data[
            "center"
        ]


        current_distance = distance(
            center,
            old_center
        )


        if (
            current_distance
            >
            MAX_IDENTITY_DISTANCE
        ):

            continue


        # ----------------------------------------------------
        # Position score
        # ----------------------------------------------------

        position_score = (
            1.0
            -
            (
                current_distance
                /
                MAX_IDENTITY_DISTANCE
            )
        )


        # ----------------------------------------------------
        # Appearance score
        # ----------------------------------------------------

        appearance_score = (
            appearance_similarity(
                appearance,
                data.get(
                    "appearance"
                )
            )
        )


        # ----------------------------------------------------
        # Size score
        # ----------------------------------------------------

        old_area = (
            data["width"]
            *
            data["height"]
        )

        new_area = (
            width
            *
            height
        )

        size_score = size_similarity(
            old_area,
            new_area
        )


        # ----------------------------------------------------
        # Direction score
        # ----------------------------------------------------

        direction_score = 0.5


        old_direction = data.get(
            "direction",
            (0, 0)
        )


        old_dx = old_direction[0]

        old_dy = old_direction[1]


        new_dx = (
            center[0]
            -
            old_center[0]
        )

        new_dy = (
            center[1]
            -
            old_center[1]
        )


        old_magnitude = math.sqrt(
            old_dx ** 2
            +
            old_dy ** 2
        )


        new_magnitude = math.sqrt(
            new_dx ** 2
            +
            new_dy ** 2
        )


        if (
            old_magnitude > 1
            and
            new_magnitude > 1
        ):

            dot = (
                old_dx * new_dx
                +
                old_dy * new_dy
            )

            cosine = (
                dot
                /
                (
                    old_magnitude
                    *
                    new_magnitude
                )
            )

            direction_score = (
                cosine + 1
            ) / 2


        # ----------------------------------------------------
        # FINAL IDENTITY SCORE
        # ----------------------------------------------------

        score = (

            APPEARANCE_WEIGHT
            *
            appearance_score

            +

            POSITION_WEIGHT
            *
            position_score

            +

            SIZE_WEIGHT
            *
            size_score

            +

            DIRECTION_WEIGHT
            *
            direction_score

        )


        # ----------------------------------------------------
        # Accept candidate
        # ----------------------------------------------------

        if score > best_score:

            best_score = score

            best_id = stable_id


    # ========================================================
    # RECONNECT
    # ========================================================

    if best_id is not None:

        stable_id = best_id

        tracker_to_stable[
            tracker_id
        ] = stable_id

        data = stable_people[
            stable_id
        ]

        data[
            "center"
        ] = center

        data[
            "width"
        ] = width

        data[
            "height"
        ] = height

        data[
            "last_seen"
        ] = current_time

        data[
            "active_tracker"
        ] = tracker_id

        if appearance is not None:

            data[
                "appearance"
            ] = appearance


        print(
            f"\n🔄 Tracker {tracker_id} "
            f"reconnected as "
            f"Person {stable_id}"
        )


        return stable_id


    # ========================================================
    # NEW PERSON
    # ========================================================

    stable_id = next_stable_id

    next_stable_id += 1


    tracker_to_stable[
        tracker_id
    ] = stable_id


    stable_people[
        stable_id
    ] = {

        "center": center,

        "width": width,

        "height": height,

        "appearance": appearance,

        "last_seen": current_time,

        "active_tracker": tracker_id,

        "direction": (0, 0)

    }


    print(
        f"\n🆕 New Person {stable_id}"
    )


    return stable_id


# ============================================================
# 19. POSTURE
# ============================================================

def determine_posture(
    keypoints
):

    try:

        ls = keypoints[5]
        rs = keypoints[6]

        lh = keypoints[11]
        rh = keypoints[12]

        lk = keypoints[13]
        rk = keypoints[14]


        if (
            ls[2] < POSE_CONFIDENCE
            or
            rs[2] < POSE_CONFIDENCE
            or
            lh[2] < POSE_CONFIDENCE
            or
            rh[2] < POSE_CONFIDENCE
        ):

            return "UNKNOWN"


        shoulder_x = (
            ls[0]
            +
            rs[0]
        ) / 2


        shoulder_y = (
            ls[1]
            +
            rs[1]
        ) / 2


        hip_x = (
            lh[0]
            +
            rh[0]
        ) / 2


        hip_y = (
            lh[1]
            +
            rh[1]
        ) / 2


        knee_y = (
            lk[1]
            +
            rk[1]
        ) / 2


        dx = (
            hip_x
            -
            shoulder_x
        )

        dy = (
            hip_y
            -
            shoulder_y
        )


        angle = abs(
            math.degrees(
                math.atan2(
                    dy,
                    dx
                )
            )
        )


        if angle > 180:

            angle = (
                360
                -
                angle
            )


        if angle < 125:

            return "SITTING"


        if angle > 150:

            return "STANDING"


        if knee_y < hip_y:

            return "SITTING"

        return "STANDING"


    except Exception:

        return "UNKNOWN"


# ============================================================
# 20. CSV
# ============================================================

csv_file = open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
)


csv_writer = csv.writer(
    csv_file
)


csv_writer.writerow([
    "Date",
    "Person ID",
    "Behaviour",
    "Posture",
    "Video Time",
    "Stationary Duration",
    "Confidence",
    "Evidence"
])


# ============================================================
# 21. START
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING STABLE ID TRACKING"
)

print(
    "=" * 70
)

print(
    "\nStable identity layer: ON"
)

print(
    "Appearance matching: ON"
)

print(
    "Position matching: ON"
)

print(
    "Size matching: ON"
)

print(
    "Direction matching: ON"
)

print(
    "\nPress Q to stop."
)


frame_number = 0

processed_frames = 0


# ============================================================
# 22. MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    # --------------------------------------------------------
    # Frame skipping
    # --------------------------------------------------------

    if (
        frame_number
        %
        FRAME_SKIP
        != 0
    ):

        out.write(
            frame
        )

        continue


    processed_frames += 1


    current_time = (
        frame_number / fps
    )


    current_video_time = (
        video_time(
            frame_number
        )
    )


    # ========================================================
    # YOLO + BOT-SORT
    # ========================================================

    try:

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[0],
            conf=DETECTION_CONFIDENCE,
            verbose=False
        )

    except Exception as e:

        print(
            "\n❌ Tracking error:"
        )

        print(e)

        break


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if len(results) > 0:

        result = results[0]

        boxes = result.boxes

        keypoints_data = (
            result.keypoints
        )


        if (
            boxes is not None
            and
            len(boxes) > 0
            and
            boxes.id is not None
        ):

            for i in range(
                len(boxes)
            ):


                # =================================================
                # TRACKER ID
                # =================================================

                tracker_id = int(
                    boxes.id[i].item()
                )


                # =================================================
                # BOX
                # =================================================

                xyxy = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )


                x1 = max(
                    0,
                    int(xyxy[0])
                )

                y1 = max(
                    0,
                    int(xyxy[1])
                )

                x2 = min(
                    frame_width - 1,
                    int(xyxy[2])
                )

                y2 = min(
                    frame_height - 1,
                    int(xyxy[3])
                )


                width = max(
                    1,
                    x2 - x1
                )

                height = max(
                    1,
                    y2 - y1
                )


                # =================================================
                # CONFIDENCE
                # =================================================

                confidence = float(
                    boxes.conf[i].item()
                )


                # =================================================
                # CENTER
                # =================================================

                center = (

                    int(
                        (x1 + x2)
                        / 2
                    ),

                    int(
                        (y1 + y2)
                        / 2
                    )

                )


                # =================================================
                # APPEARANCE
                # =================================================

                appearance = (
                    get_appearance(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )


                # =================================================
                # STABLE PERSON ID
                # =================================================

                stable_id = (
                    get_stable_id(
                        tracker_id,
                        center,
                        width,
                        height,
                        appearance,
                        current_time
                    )
                )


                # =================================================
                # MOVEMENT
                # =================================================

                positions = (
                    position_history[
                        stable_id
                    ]
                )


                if positions:

                    move_distance = (
                        distance(
                            positions[-1],
                            center
                        )
                    )

                else:

                    move_distance = 0


                positions.append(
                    center
                )


                if len(
                    positions
                ) > MOVEMENT_HISTORY_FRAMES:

                    positions.pop(0)


                if (
                    move_distance
                    >=
                    MOVEMENT_DISTANCE_THRESHOLD
                ):

                    movement = "MOVING"

                else:

                    movement = "STATIONARY"


                movement_history[
                    stable_id
                ].append(
                    movement
                )


                if len(
                    movement_history[
                        stable_id
                    ]
                ) > MOVEMENT_HISTORY_FRAMES:

                    movement_history[
                        stable_id
                    ].pop(0)


                smoothed_movement = (
                    smooth_value(
                        movement_history[
                            stable_id
                        ]
                    )
                )


                # =================================================
                # POSTURE
                # =================================================

                posture = "UNKNOWN"


                if (
                    keypoints_data
                    is not None
                    and
                    len(
                        keypoints_data
                    ) > i
                ):

                    try:

                        kp = (
                            keypoints_data
                            .xy[i]
                            .cpu()
                            .numpy()
                        )


                        kp_conf = (
                            keypoints_data
                            .conf[i]
                            .cpu()
                            .numpy()
                        )


                        combined = []


                        for j in range(
                            len(kp)
                        ):

                            combined.append([
                                kp[j][0],
                                kp[j][1],
                                kp_conf[j]
                            ])


                        posture = (
                            determine_posture(
                                combined
                            )
                        )


                    except Exception:

                        posture = "UNKNOWN"


                posture_history[
                    stable_id
                ].append(
                    posture
                )


                if len(
                    posture_history[
                        stable_id
                    ]
                ) > POSTURE_HISTORY_FRAMES:

                    posture_history[
                        stable_id
                    ].pop(0)


                valid_postures = [

                    p

                    for p in
                    posture_history[
                        stable_id
                    ]

                    if p != "UNKNOWN"

                ]


                if valid_postures:

                    smoothed_posture = (
                        smooth_value(
                            valid_postures
                        )
                    )

                else:

                    smoothed_posture = (
                        "UNKNOWN"
                    )


                # =================================================
                # STATIONARY TIMER
                # =================================================

                if (
                    smoothed_movement
                    ==
                    "STATIONARY"
                ):

                    if (
                        stable_id
                        not in
                        stationary_start_time
                    ):

                        stationary_start_time[
                            stable_id
                        ] = current_time


                    stationary_duration = (

                        current_time
                        -
                        stationary_start_time[
                            stable_id
                        ]

                    )

                else:

                    stationary_start_time.pop(
                        stable_id,
                        None
                    )

                    stationary_duration = 0

                    alert_saved.discard(
                        stable_id
                    )

                    event_logged.discard(
                        stable_id
                    )


                # =================================================
                # BEHAVIOUR
                # =================================================

                if (
                    smoothed_movement
                    ==
                    "MOVING"
                ):

                    if (
                        smoothed_posture
                        ==
                        "STANDING"
                    ):

                        behaviour = "Walking"

                    elif (
                        smoothed_posture
                        ==
                        "SITTING"
                    ):

                        behaviour = (
                            "Moving / Transitioning"
                        )

                    else:

                        behaviour = "Moving"


                else:

                    if (
                        smoothed_posture
                        ==
                        "SITTING"
                    ):

                        behaviour = "Sitting"

                    elif (
                        smoothed_posture
                        ==
                        "STANDING"
                    ):

                        behaviour = (
                            "Standing still"
                        )

                    else:

                        behaviour = "Stationary"


                # =================================================
                # ABNORMAL
                # =================================================

                abnormal = (

                    smoothed_movement
                    ==
                    "STATIONARY"

                    and

                    stationary_duration
                    >=
                    STATIONARY_THRESHOLD

                )


                # =================================================
                # COLOR
                # =================================================

                if abnormal:

                    box_color = (
                        0,
                        0,
                        255
                    )

                elif (
                    smoothed_movement
                    ==
                    "STATIONARY"
                ):

                    box_color = (
                        0,
                        255,
                        255
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )


                # =================================================
                # PERSON BOX
                # =================================================

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        y1
                    ),
                    (
                        x2,
                        y2
                    ),
                    box_color,
                    3
                )


                # =================================================
                # STABLE ID
                # =================================================

                cv2.putText(
                    frame,
                    f"Person {stable_id}",
                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    box_color,
                    2
                )


                # =================================================
                # BEHAVIOUR
                # =================================================

                cv2.putText(
                    frame,
                    (
                        f"{behaviour} | "
                        f"{smoothed_posture}"
                    ),
                    (
                        x1,
                        min(
                            frame_height - 10,
                            y2 + 25
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    box_color,
                    2
                )


                # =================================================
                # TIMER
                # =================================================

                if (
                    smoothed_movement
                    ==
                    "STATIONARY"
                ):

                    cv2.putText(
                        frame,
                        (
                            f"Stationary "
                            f"{stationary_duration:.1f}s"
                        ),
                        (
                            x1,
                            min(
                                frame_height - 10,
                                y2 + 48
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        box_color,
                        2
                    )


                # =================================================
                # ABNORMAL ALERT
                # =================================================

                if abnormal:

                    cv2.rectangle(
                        frame,
                        (
                            10,
                            10
                        ),
                        (
                            frame_width - 10,
                            65
                        ),
                        (
                            0,
                            0,
                            255
                        ),
                        -1
                    )


                    cv2.putText(
                        frame,
                        (
                            f"ALERT: Person "
                            f"{stable_id} "
                            f"stationary "
                            f"{stationary_duration:.1f}s"
                        ),
                        (
                            25,
                            48
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (
                            255,
                            255,
                            255
                        ),
                        2
                    )


                    # =================================================
                    # EVIDENCE
                    # =================================================

                    if (
                        stable_id
                        not in
                        alert_saved
                    ):

                        evidence_name = (

                            f"person_"
                            f"{stable_id}_"
                            f"time_"
                            f"{current_video_time.replace(':', '-')}"
                            f".jpg"

                        )


                        evidence_path = (
                            os.path.join(
                                EVIDENCE_FOLDER,
                                evidence_name
                            )
                        )


                        crop = frame[
                            y1:y2,
                            x1:x2
                        ]


                        if crop.size > 0:

                            cv2.imwrite(
                                evidence_path,
                                crop
                            )

                        else:

                            evidence_path = ""


                        alert_saved.add(
                            stable_id
                        )


                        # =================================================
                        # CSV
                        # =================================================

                        if (
                            stable_id
                            not in
                            event_logged
                        ):

                            csv_writer.writerow([

                                datetime.now().strftime(
                                    "%Y-%m-%d"
                                ),

                                stable_id,

                                behaviour,

                                smoothed_posture,

                                current_video_time,

                                round(
                                    stationary_duration,
                                    2
                                ),

                                round(
                                    confidence,
                                    3
                                ),

                                evidence_path

                            ])


                            csv_file.flush()


                            event_logged.add(
                                stable_id
                            )


    # ========================================================
    # TOP PANEL
    # ========================================================

    cv2.rectangle(
        frame,
        (
            0,
            0
        ),
        (
            frame_width,
            105
        ),
        (
            20,
            20,
            20
        ),
        -1
    )


    cv2.putText(
        frame,
        "HACKNEX PS07",
        (
            20,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        frame,
        "STABLE IDENTITY + ReID",
        (
            20,
            58
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        frame,
        f"Time: {current_video_time}",
        (
            20,
            87
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            255,
            255,
            255
        ),
        2
    )


    # ========================================================
    # LEGEND
    # ========================================================

    cv2.putText(
        frame,
        "GREEN = MOVING",
        (
            20,
            135
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            0,
            255,
            0
        ),
        2
    )


    cv2.putText(
        frame,
        "YELLOW = STATIONARY",
        (
            20,
            160
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            0,
            255,
            255
        ),
        2
    )


    cv2.putText(
        frame,
        "RED = ABNORMAL",
        (
            20,
            185
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            0,
            0,
            255
        ),
        2
    )


    # ========================================================
    # SAVE
    # ========================================================

    out.write(
        frame
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "HACKNEX PS07 - Stable Person Tracking",
        frame
    )


    # ========================================================
    # Q TO STOP
    # ========================================================

    if (
        cv2.waitKey(1)
        &
        0xFF
    ) == ord("q"):

        print(
            "\nStopping..."
        )

        break


# ============================================================
# 23. CLEANUP
# ============================================================

cap.release()

out.release()

csv_file.close()

cv2.destroyAllWindows()


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "✅ ANALYSIS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nInput:"
)

print(
    VIDEO_PATH
)

print(
    "\nOutput:"
)

print(
    OUTPUT_VIDEO
)

print(
    "\nEvidence:"
)

print(
    EVIDENCE_FOLDER
)

print(
    "\nCSV:"
)

print(
    CSV_FILE
)

print(
    "\nStable persons created:"
)

print(
    next_stable_id - 1
)

print(
    "\n" + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)

input(
    "\nPress ENTER to close..."
)