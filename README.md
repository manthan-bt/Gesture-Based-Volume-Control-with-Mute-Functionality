# Gesture-Based Volume Control with Mute Functionality

A real-time touchless volume control system built using computer vision. Control your system volume using hand gestures — no keyboard, no mouse required.

Built as part of the Computer Vision and Image Processing (CVIP) course at PES University.

---

## Demo

| Open Hand (100%) | Pinch Gesture (55%) | Confirm Gesture | Fist (Mute) |
|:---:|:---:|:---:|:---:|
| ![Open Hand](Screenshot%202025-04-16%20004448.png) | ![Pinch](Screenshot%202025-04-16%20004541.png) | ![Confirm](Screenshot%202025-04-16%20004605.png) | ![Mute](Screenshot%202025-04-16%20004512.png) |

---

## How It Works

MediaPipe detects **21 hand landmarks** in real time via webcam. The Euclidean distance between the **thumb tip** and **index finger tip** is linearly mapped to system volume (0–100%) using NumPy interpolation. Volume is committed to the system via PyCaw only when the **pinky finger is down**, preventing accidental changes.

### Gesture Reference

| Gesture | Action |
|---|---|
| Thumb-index spread | Adjusts volume proportionally to distance |
| Pinky down | Confirms and sets the volume on the system |
| Pinky up | Preview mode — volume not committed |
| Fist (all fingers closed) | Instantly mutes the system |

Volume transitions are smoothed using rounded step interpolation to avoid jitter during rapid hand movement.

---

## Project Structure

```
├── HandTrackingModule.py       # Hand detection, 21-landmark extraction, finger state logic, distance calculation
├── CVIP_project.py             # Main app — volume mapping, gesture recognition, PyCaw integration
├── CVIP_Project_PES1UG22EC907_EC131.pdf    # Code submission report
├── CVIP_project.pdf            # Project presentation slides
└── Screenshot *.png            # Demo screenshots
```

---

## Installation

> **Note:** PyCaw is Windows-only. This project runs on Windows with a webcam.

**Install dependencies:**

```bash
pip install opencv-python mediapipe numpy pycaw comtypes
```

**Run the project:**

```bash
python CVIP_project.py
```

Press `q` to quit.

---

## Tech Stack

| Library | Purpose |
|---|---|
| OpenCV | Webcam capture and real-time frame rendering |
| MediaPipe | 21-point hand landmark detection |
| NumPy | Distance interpolation and volume percentage mapping |
| PyCaw | Windows Core Audio API for system volume control |

---

## Authors

**Manthan B T** — [github.com/manthan-bt](https://github.com/manthan-bt)  
**Krutharth M C**

PES University, Bengaluru — Department of ECE  
Course Instructor: Prof. Vanamala H R
