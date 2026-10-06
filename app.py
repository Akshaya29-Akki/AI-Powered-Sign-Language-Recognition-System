import cv2
import pickle
import numpy as np

with open("sign_language_model.pkl", "rb") as file:
    model = pickle.load(file)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    input("Press Enter to close...")
    exit()

print("AI Sign Language Recognition started!")
print("Camera opened successfully!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    x1 = int(width * 0.25)
    y1 = int(height * 0.15)
    x2 = int(width * 0.75)
    y2 = int(height * 0.85)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    features = normalized.flatten().reshape(1, -1)

    prediction = model.predict(features)[0]

    cv2.putText(
        frame,
        "Sign: " + str(prediction),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Powered Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Application closed.")