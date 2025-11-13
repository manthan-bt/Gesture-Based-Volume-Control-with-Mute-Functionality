import cv2
import time
import numpy as np
import HandTrackingModule as htm  # Custom module for hand tracking
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # For system volume control

# Webcam frame dimensions
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height

pTime = 0  # Previous time for FPS calculation
detector = htm.handDetector(detectionCon=0.7, maxHands=1)  # Initialize hand detector

# Initialize volume control using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # Get system volume range
minVol, maxVol = volRange[0], volRange[1]
volBar = 400  # Position of volume bar
volPer = 0  # Volume percentage
colorVol = (255, 0, 0)  # Initial color of volume text/bar
smoothness = 5  # Smooth step size for volume rounding

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands and landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)

    if lmList and bbox:
        # Calculate hand bounding box area to filter out too small or too large detections
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

        if 250 < area < 1000:  # Process only valid hand sizes
            fingers = detector.fingersUp()  # Get finger states (1 = up, 0 = down)

            # If all fingers are down (fist gesture) → mute
            if fingers == [0, 0, 0, 0, 0]:
                volume.SetMasterVolumeLevelScalar(0.0, None)  # Mute
                cv2.putText(img, "Muted (Fist)", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                colorVol = (0, 0, 255)  # Red for mute

            else:
                # Measure distance between thumb tip (id 4) and index finger tip (id 8)
                length, img, lineInfo = detector.findDistance(4, 8, img)
                
                # Map the distance to volume range (bar position and percentage)
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])
                volPer = smoothness * round(volPer / smoothness)  # Apply smooth stepping

                # Confirm volume set only when pinky is down
                if not fingers[4]:  # If pinky is down
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cx, cy = lineInfo[4], lineInfo[5]  # Center point between thumb and index
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)  # Green circle on confirmation
                    colorVol = (0, 255, 0)  # Green for confirmed volume
                else:
                    colorVol = (255, 0, 0)  # Blue when waiting for confirmation

    # Draw volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # Outline
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # Filled bar
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display current system volume set
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Show final output
    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()