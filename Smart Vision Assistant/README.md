# Smart Vision Assistant

**AI-powered stereo vision system for the visually impaired** — real-time object detection with distance estimation, OCR-based text reading, and voice alerts, built on a dual-camera (ESP32-CAM / Raspberry Pi) setup.

---

## Overview

Smart Vision Assistant turns a pair of cameras into a wearable navigation aid. It fuses **stereo depth estimation** with **YOLO object detection** and **OCR** to tell the user, out loud, what's around them and how far away it is.

- **Object Detection** — YOLOv8/YOLO11 identifies obstacles and objects in the scene
- **Distance Estimation** — stereo disparity mapping (OpenCV SGBM) computes real-world distance to each detected object
- **Voice Alerts** — text-to-speech (`pyttsx3`) speaks out the closest/detected objects and their distance
- **Text Reading (OCR)** — `EasyOCR` reads signs, labels, and handwritten text aloud
- **Dual Camera Support** — works with either two **ESP32-CAM** modules (WiFi stream) or a dual **Raspberry Pi Camera** rig

---

## Demo

| Object Detection & Depth | OCR Text Reading |
|:---:|:---:|
| ![Object detection and depth estimation](../Hardware%20Demo%20Videos%20and%20photos-20260717T165354Z-1-001/Hardware%20Demo%20Videos%20and%20photos/Object%20detection%20and%20depth%20estimation%20using%20dual%20cam%20.jpeg) | ![OCR testing result](../Hardware%20Demo%20Videos%20and%20photos-20260717T165354Z-1-001/Hardware%20Demo%20Videos%20and%20photos/OCR%20testing%20result.jpeg) |

| Stereo Calibration | Handwritten Text Detection |
|:---:|:---:|
| ![Stereo calibration using chessboard](../Hardware%20Demo%20Videos%20and%20photos-20260717T165354Z-1-001/Hardware%20Demo%20Videos%20and%20photos/Stereo%20Callibration%20using%20Chessboard%20image.jpeg) | ![Handwritten text detection using OCR](../Hardware%20Demo%20Videos%20and%20photos-20260717T165354Z-1-001/Hardware%20Demo%20Videos%20and%20photos/Handwritten%20text%20detection%20using%20ocr.jpeg) |

A full demo video is available in the `Hardware Demo Videos and photos` folder.

---

## Architecture

### System Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              HARDWARE LAYER                              │
│                                                                            │
│   ESP32-CAM (Left)      ESP32-CAM (Right)      — or —      Pi Camera x2  │
│   MJPEG @ 192.168.x.x   MJPEG @ 192.168.x.x            (picamera2 lib)   │
└───────────────┬─────────────────────┬────────────────────────┬──────────┘
                │  HTTP MJPEG stream  │                        │ CSI ribbon
                ▼                     ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         CAPTURE / ACQUISITION LAYER                      │
│  esp32_cam.py → CameraStreamThread (x2, daemon threads, JPEG frame       │
│                 parsing from byte stream, mirror-flip correction)        │
│  raspberry_picam.py → Picamera2.capture_array() (x2, synchronous)        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      CALIBRATION / RECTIFICATION LAYER                   │
│  stereo_calib.xml / stereo_calib2.xml  (K1, D1, K2, D2, R, T, R1, R2,    │
│  P1, P2, Q — generated offline by callibration_file_*.py)                │
│  cv2.initUndistortRectifyMap()  →  cv2.remap()  (undistort + align L/R)  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          PERCEPTION LAYER (parallel)                     │
│                                                                            │
│  ┌────────────────────────────┐   ┌────────────────────────────────────┐ │
│  │  STEREO DEPTH PIPELINE     │   │  OBJECT DETECTION PIPELINE         │ │
│  │  grayL/grayR → SGBM        │   │  left frame → YOLOv8n / YOLO11n    │ │
│  │  left+right matcher (WLS   │   │  → bounding boxes + class labels   │ │
│  │  filtered, esp32 variant)  │   │  → confidence threshold filter     │ │
│  │  → disparity map           │   │                                    │ │
│  │  → reprojectImageTo3D(Q)   │   └──────────────┬─────────────────────┘ │
│  │  → per-pixel depth (X,Y,Z) │                  │                       │
│  └──────────────┬─────────────┘                  │                       │
│                 │                                 │                       │
│                 └───────────────┬─────────────────┘                       │
│                                 ▼                                         │
│                  FUSION: crop center 50% ROI of each detected             │
│                  box → sample depth map → filter outliers/NaNs →          │
│                  median/mean distance per object (meters)                │
│                                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  OCR PIPELINE (independent, own thread/timer)                      │  │
│  │  periodic frame grab → grayscale → EasyOCR.readtext()              │  │
│  │  → paragraph text → dedupe vs. last reading                        │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          DECISION / ALERT LAYER                          │
│  • Pick closest / relevant object → build alert string                  │
│  • Per-label audio cooldown (avoid repeat spam)                         │
│  • Queue text to the audio worker (non-blocking)                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT LAYER                                │
│  pyttsx3 TTS engine (dedicated thread, queue-driven)                    │
│  cv2.imshow — live annotated feed + disparity/depth color map           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Concurrency Model

Both variants keep the main OpenCV/YOLO loop from blocking on slow I/O by isolating work into daemon threads:

| Thread | Responsibility | Communication |
|---|---|---|
| **Main loop** | Frame rectification, stereo matching, YOLO inference, ROI depth fusion, UI rendering | — |
| **Camera stream thread(s)** *(ESP32 variant)* | Read raw MJPEG bytes off the socket, extract JPEG frames, decode, flip | Shared `self.frame` (last-write-wins) |
| **OCR worker thread** | Run EasyOCR on the latest submitted frame, independent of the detection cadence | `pending_frame` handoff (ESP32) / time-gated call in main loop (Pi) |
| **Audio worker thread** | Own a single `pyttsx3` engine instance, speak queued text sequentially | `text_queue` (ESP32) / `queue.Queue` (Pi) |

This prevents TTS playback or OCR inference (both relatively slow) from stalling the real-time video/detection loop.

### Two Hardware Variants

| | `esp32_cam.py` | `raspberry_picam.py` |
|---|---|---|
| Camera input | WiFi MJPEG stream (`urllib.request`) | `picamera2` CSI camera API |
| Depth pipeline | SGBM + WLS-filtered disparity, colorized depth map window | SGBM + histogram equalization + median blur |
| Detection model | YOLOv8n | YOLO11n |
| OCR cadence | Every detection cycle (2s), async thread | Fixed interval (`OCR_INTERVAL`), inline in main loop |
| Audio | Custom `AudioAlertWorker` class | `queue.Queue` + worker function |
| Calibration file | `stereo_calib2.xml` | `stereo_calib.xml` |

### Data Flow Summary

1. **Capture** — two synchronized camera feeds (left/right) are streamed in
2. **Rectify** — frames are undistorted and aligned using pre-computed stereo calibration matrices
3. **Detect** — YOLO scans the left frame for objects
4. **Measure** — for each detected object, the disparity map is used to compute real-world distance (in meters)
5. **Speak** — the closest/relevant object and its distance are announced via text-to-speech
6. **Read** — periodically, OCR scans the frame for text and reads it aloud

---

## Project Structure

```
Smart Vision Assistant/
├── esp32_cam.py                   # Main app — dual ESP32-CAM (WiFi stream) version
├── raspberry_picam.py             # Main app — dual Raspberry Pi Camera version
├── callibration_file_esp32cam.py  # Stereo calibration script for ESP32-CAM setup
└── callibration_file_raspcam.py   # Stereo calibration script for Raspberry Pi Camera setup
```

---

## Requirements

- Python 3.9+
- Two cameras: either
  - two **ESP32-CAM** modules configured as MJPEG streaming servers on the same network, **or**
  - a **Raspberry Pi** with two cameras (via `picamera2`)
- A printed chessboard pattern for calibration

### Install dependencies

```bash
pip install opencv-contrib-python numpy ultralytics easyocr pyttsx3
```

> For the Raspberry Pi variant, also install `picamera2` (usually preinstalled on Raspberry Pi OS).

---

## Getting Started

### 1. Calibrate your stereo camera pair

Capture chessboard images from both cameras and place them in the expected folders, then run the calibration script matching your hardware:

```bash
# ESP32-CAM setup (expects StereoImages/left, StereoImages/right)
python callibration_file_esp32cam.py

# Raspberry Pi Camera setup (expects left_frames, right_frames)
python callibration_file_raspcam.py
```

This generates `stereo_calib2.xml` or `stereo_calib.xml` — required by the main app.

### 2. Run the assistant

```bash
# ESP32-CAM version — update the stream URLs (urlL / urlR) to match your devices' IPs
python esp32_cam.py

# Raspberry Pi Camera version
python raspberry_picam.py
```

Press **`q`** to exit.

---

## Configuration Notes

- **ESP32-CAM IPs**: update `urlL` and `urlR` in `esp32_cam.py` to your cameras' local network addresses.
- **Detection confidence / frame skip / OCR interval**: tunable constants near the top of `raspberry_picam.py` (`CONFIDENCE`, `FRAME_SKIP`, `OCR_INTERVAL`, `AUDIO_COOLDOWN`).
- **YOLO model weights** (`yolov8n.pt` / `yolo11n.pt`) are auto-downloaded by `ultralytics` on first run.

---

## Documentation

A detailed project report is included at the repository root: `SMART VISION ASSISTANCE SYSTEM FOR VISUALLY IMPAIRED.pdf`.

---

## Motivation

Built to give visually impaired users real-time spatial and textual awareness of their surroundings — combining obstacle distance alerts with the ability to "read" printed or handwritten text, using low-cost, accessible hardware.
