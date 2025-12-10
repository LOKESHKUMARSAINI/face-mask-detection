import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)
        if results.detections:
            for det in results.detections:
                mp_drawing.draw_detection(frame, det)
        cv2.imshow("Face Mask Demo (placeholder)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
