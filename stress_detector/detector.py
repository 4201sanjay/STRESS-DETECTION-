import cv2
import time
import threading
import requests
import os
from dotenv import load_dotenv

load_dotenv()

model_id = os.getenv('MODEL_ID_1') 
API_KEY = os.getenv('API_KEY')

face_image_path = "captured_face.png"
last_captured_time = 0  
stress_prediction = {} 

def capture_face(frame, gray, faces):

    if len(faces) > 0:

        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        cv2.imwrite(face_image_path, face)

        threading.Thread(target=call_roboflow_api, args=(face_image_path,)).start()

def call_roboflow_api(image_path):

    global stress_prediction  

    with open(image_path, "rb") as image_file:
        
        response = requests.post(
            f"https://detect.roboflow.com/{model_id}?api_key={API_KEY}",
            files={"file": image_file}
        )

    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:", result)
        stress_prediction = result['predictions']
        print("stress", stress_prediction)
    else:
        print(f"Error: {response.status_code}, {response.text}")

def main():

    global last_captured_time

    cap = cv2.VideoCapture(0)
    time.sleep(2)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
            if stress_prediction:
                predicted_class = max(stress_prediction, key=lambda k: stress_prediction[k]['confidence'])
                confidence = stress_prediction[predicted_class]['confidence']
                # predicted_class = stress_prediction[0]['class']
                # confidence = stress_prediction[0]['confidence']
                text = f"{predicted_class}: {confidence:.2f}"

        
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Live Camera', frame)

        current_time = time.time()
        if current_time - last_captured_time >= 10:
            capture_face(frame, gray, faces)
            last_captured_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
