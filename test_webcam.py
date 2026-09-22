import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print("Left hand detected:", results.left_hand_landmarks is not None)
        cv2.imshow("Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()