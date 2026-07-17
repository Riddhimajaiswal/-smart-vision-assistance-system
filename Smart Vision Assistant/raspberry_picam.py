import cv2
import numpy as np
import time
import threading
import queue
import pyttsx3

from ultralytics import YOLO
from picamera2 import Picamera2
import easyocr

# ==========================================================
# CONFIGURATION
# ==========================================================

WIDTH = 640
HEIGHT = 480

YOLO_SIZE = 320
CONFIDENCE = 0.50

FRAME_SKIP = 5
OCR_INTERVAL = 10

AUDIO_COOLDOWN = 5

# ==========================================================
# AUDIO
# ==========================================================

speech_queue = queue.Queue()


def audio_worker():

    engine = pyttsx3.init()

    engine.setProperty("rate", 160)

    while True:

        text = speech_queue.get()

        if text is None:
            break

        engine.say(text)

        engine.runAndWait()

        speech_queue.task_done()


threading.Thread(
    target=audio_worker,
    daemon=True
).start()


def speak(text):

    speech_queue.put(text)


# ==========================================================
# LOAD YOLO
# ==========================================================

print("\nLoading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO Loaded")


# ==========================================================
# LOAD OCR
# ==========================================================

print("Loading OCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("OCR Loaded")


# ==========================================================
# LOAD CALIBRATION
# ==========================================================

print("\nLoading Stereo Calibration...")

fs = cv2.FileStorage(
    "stereo_calib.xml",
    cv2.FILE_STORAGE_READ
)

K1 = fs.getNode("K1").mat()
D1 = fs.getNode("D1").mat()

K2 = fs.getNode("K2").mat()
D2 = fs.getNode("D2").mat()

R1 = fs.getNode("R1").mat()
R2 = fs.getNode("R2").mat()

P1 = fs.getNode("P1").mat()
P2 = fs.getNode("P2").mat()

Q = fs.getNode("Q").mat()

fs.release()

if Q is None:
    raise RuntimeError("stereo_calib.xml not found!")

print("Calibration Loaded Successfully")

# ==========================================================
# RECTIFICATION MAPS
# ==========================================================

print("Creating Rectification Maps...")

leftMapX, leftMapY = cv2.initUndistortRectifyMap(
    K1,
    D1,
    R1,
    P1,
    (WIDTH, HEIGHT),
    cv2.CV_32FC1
)

rightMapX, rightMapY = cv2.initUndistortRectifyMap(
    K2,
    D2,
    R2,
    P2,
    (WIDTH, HEIGHT),
    cv2.CV_32FC1
)

print("Rectification Maps Created")

# ==========================================================
# CAMERA SETUP
# ==========================================================

print("Opening Cameras...")

left_cam = Picamera2(0)
right_cam = Picamera2(1)

configL = left_cam.create_preview_configuration(
    main={"size": (WIDTH, HEIGHT)}
)

configR = right_cam.create_preview_configuration(
    main={"size": (WIDTH, HEIGHT)}
)

left_cam.configure(configL)
right_cam.configure(configR)

left_cam.start()
right_cam.start()

time.sleep(2)

print("Both Cameras Started")

# ==========================================================
# STEREO MATCHER
# ==========================================================

print("Initializing StereoSGBM...")

stereo = cv2.StereoSGBM_create(

    minDisparity=0,

    numDisparities=128,

    blockSize=7,

    P1=8 * 3 * 7 * 7,

    P2=32 * 3 * 7 * 7,

    uniquenessRatio=8,

    speckleWindowSize=120,

    speckleRange=32,

    disp12MaxDiff=1,

    preFilterCap=31,

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY

)

print("Stereo Matcher Ready")

# ==========================================================
# VARIABLES
# ==========================================================

frame_count = 0
fps_counter = 0
fps = 0
fps_timer = time.time()

last_boxes = []

last_text = ""
last_ocr = 0

last_audio = {}

print("\n=====================================")
print(" SMART VISION SYSTEM STARTED ")
print(" Press Q to Exit ")
print("=====================================")

# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    # ------------------------------------------------------
    # Capture Frames
    # ------------------------------------------------------

    frameL = left_cam.capture_array()
    frameR = right_cam.capture_array()

    frameL = cv2.cvtColor(frameL, cv2.COLOR_RGB2BGR)
    frameR = cv2.cvtColor(frameR, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------
    # Stereo Rectification
    # ------------------------------------------------------

    rectL = cv2.remap(
        frameL,
        leftMapX,
        leftMapY,
        cv2.INTER_LINEAR
    )

    rectR = cv2.remap(
        frameR,
        rightMapX,
        rightMapY,
        cv2.INTER_LINEAR
    )

    # ------------------------------------------------------
    # Convert to Gray
    # ------------------------------------------------------

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------
    # Histogram Equalization
    # Improves disparity quality
    # ------------------------------------------------------

    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    # ------------------------------------------------------
    # Compute Disparity
    # ------------------------------------------------------

    disparity = stereo.compute(
        grayL,
        grayR
    ).astype(np.float32)

    disparity /= 16.0

    disparity[disparity < 0] = 0

    disparity = cv2.medianBlur(disparity,5)

    # ------------------------------------------------------
    # Convert Disparity -> 3D
    # ------------------------------------------------------

    points3D = cv2.reprojectImageTo3D(
        disparity,
        Q
    )

    frame_count += 1
    fps_counter += 1

    # ------------------------------------------------------
    # YOLO Detection
    # ------------------------------------------------------

    if frame_count % FRAME_SKIP == 0:

        results = model.predict(
            rectL,
            imgsz=YOLO_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        current_boxes = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cls = int(box.cls[0])

                label = model.names[cls]

                x1 = max(0,x1)
                y1 = max(0,y1)

                x2 = min(WIDTH-1,x2)
                y2 = min(HEIGHT-1,y2)

                distance_text = "Unknown"

                audio_text = ""

                # ==========================================
                # CENTER 50% ROI
                # ==========================================

                bw = x2-x1
                bh = y2-y1

                cx1 = x1 + int(bw*0.25)
                cx2 = x2 - int(bw*0.25)

                cy1 = y1 + int(bh*0.25)
                cy2 = y2 - int(bh*0.25)

                roi3D = points3D[
                    cy1:cy2,
                    cx1:cx2
                ]

                roiDisp = disparity[
                    cy1:cy2,
                    cx1:cx2
                ]

                if roi3D.size > 0:

                    X = roi3D[:,:,0].flatten()
                    Y = roi3D[:,:,1].flatten()
                    Z = roi3D[:,:,2].flatten()

                    valid = np.isfinite(Z)

                    valid &= (Z > 0.10)

                    valid &= (Z < 8)

                    valid &= (roiDisp.flatten() > 2)

                    if np.count_nonzero(valid) > 100:

                        X = np.median(X[valid])
                        Y = np.median(Y[valid])
                        Z = np.median(Z[valid])

                        distance = np.sqrt(
                            X*X +
                            Y*Y +
                            Z*Z
                        )

                        distance_text = f"{distance:.2f} m"

                        audio_text = (
                            f"{label} detected at "
                            f"{distance:.1f} meters"
                        )

                current_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        label,
                        distance_text
                    )
                )

                now = time.time()

                if audio_text != "":

                    if (
                        label not in last_audio
                        or
                        now-last_audio[label] > AUDIO_COOLDOWN
                    ):

                        print(audio_text)

                        speak(audio_text)

                        last_audio[label] = now

        last_boxes = current_boxes
            # ==========================================================
    # DRAW BOUNDING BOXES
    # ==========================================================

    for (x1, y1, x2, y2, label, distance_text) in last_boxes:

        cv2.rectangle(
            rectL,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            rectL,
            f"{label}  {distance_text}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )

    # ==========================================================
    # OCR
    # ==========================================================

    if time.time() - last_ocr > OCR_INTERVAL:

        try:

            texts = reader.readtext(
                rectL,
                detail=0,
                paragraph=True
            )

            if len(texts):

                detected_text = " ".join(texts)

                detected_text = detected_text.strip()

                if detected_text != "" and detected_text != last_text:

                    print("\nOCR :", detected_text)

                    speak("Reading text. " + detected_text)

                    last_text = detected_text

        except Exception as e:

            print("OCR Error:", e)

        last_ocr = time.time()

    # ==========================================================
    # FPS CALCULATION
    # ==========================================================

    if time.time() - fps_timer >= 1:

        fps = fps_counter

        fps_counter = 0

        fps_timer = time.time()

    cv2.putText(
        rectL,
        f"FPS : {fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # ==========================================================
    # OPTIONAL DISPARITY MAP
    # Press D to Show/Hide
    # ==========================================================

    disparity_display = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    disparity_display = np.uint8(disparity_display)

    # ==========================================================
    # DISPLAY WINDOWS
    # ==========================================================

    cv2.imshow(
        "Smart Vision",
        rectL
    )

    cv2.imshow(
        "Disparity Map",
        disparity_display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):

        break
    # ==========================================================
# CLEANUP
# ==========================================================

print("\nClosing Smart Vision...")

speech_queue.put(None)

left_cam.stop()

right_cam.stop()

cv2.destroyAllWindows()

print("Done.")