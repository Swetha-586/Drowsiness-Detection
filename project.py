import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)

closed_eyes_frame = 0
alert_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Check for drowsiness
    if len(eyes) >= 2:
        closed_eyes_frame = 0
        alert_triggered = False
        cv2.putText(frame, "You are Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        closed_eyes_frame += 1
        if closed_eyes_frame > 20:
            cv2.putText(frame, "Drowsiness Detected! Wake up!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not alert_triggered:
                engine.say("Alert! Wake up!")
                engine.runAndWait()
                alert_triggered = True

    # Show video frame
    cv2.imshow('Drowsiness Detection - Eyes Only', frame)

    # Break loop if 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()

