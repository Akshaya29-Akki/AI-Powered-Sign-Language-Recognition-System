import cv2
import os
import time

# Signs to collect
SIGNS = ["HELLO", "YES", "NO", "THANK_YOU"]

# Create dataset folders
for sign in SIGNS:
    os.makedirs(os.path.join("dataset", sign), exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    print("Check that your camera is connected and camera permission is enabled.")
    exit()

print("Camera opened successfully!")
print("Press H = HELLO")
print("Press Y = YES")
print("Press N = NO")
print("Press T = THANK YOU")
print("Press Q = Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "H: HELLO | Y: YES | N: NO | T: THANK YOU | Q: QUIT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Sign Language - Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    sign = None

    if key == ord("h"):
        sign = "HELLO"
    elif key == ord("y"):
        sign = "YES"
    elif key == ord("n"):
        sign = "NO"
    elif key == ord("t"):
        sign = "THANK_YOU"
    elif key == ord("q"):
        break

    if sign:
        filename = os.path.join(
            "dataset",
            sign,
            f"{int(time.time() * 1000)}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {sign}")

cap.release()
cv2.destroyAllWindows()