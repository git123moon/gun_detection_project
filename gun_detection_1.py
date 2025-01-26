import numpy as np
import cv2
import imutils
import os

cascade_path = r'D:\semester 7\Python_projects\gun_detectipon_systen\.venv\cascade.xml'
if not os.path.exists(cascade_path):
    print(f"Error: The file '{cascade_path}' does not exist!")
    exit()

gun_cascade = cv2.CascadeClassifier(cascade_path)

if gun_cascade.empty():
    print("Error loading cascade.xml!")
    exit()
else:
    print("cascade.xml loaded successfully.")

camera = cv2.VideoCapture(0)
gun_exist = False  # Initialize gun detection status

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for better processing speed
    frame = imutils.resize(frame, width=500)

    # Convert frame to grayscale for cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame
    gun = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # If any guns are detected, set the flag and print detection result
    if len(gun) > 0:
        gun_exist = True
        print("Gun detected!")
    else:
        gun_exist = False

    # Draw rectangles around detected guns
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the feed
    cv2.imshow("Security Feed", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()

# Output detection result after the session
if gun_exist:
    print("Guns detected during the session.")
else:
    print("No guns detected during the session.")
