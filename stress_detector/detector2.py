import cv2
import time
import threading
import os
from google.cloud import vision
from google.oauth2 import service_account

# Path to your service account key JSON file
service_account_json = "credential.json"

# Authenticate using the service account JSON
credentials = service_account.Credentials.from_service_account_file(service_account_json)

# Initialize the Vision API client
client = vision.ImageAnnotatorClient(credentials=credentials)

# Variable to store the face image path
face_image_path = "captured_face.png"
last_captured_time = 0  
emotion_prediction = {}

def capture_face(frame, gray, faces):
    if len(faces) > 0:
        # Crop the first detected face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        # Save the captured face image temporarily
        cv2.imwrite(face_image_path, face)

        # Call the Google Vision API in a separate thread to avoid blocking the live feed
        threading.Thread(target=call_google_vision_api, args=(face_image_path,)).start()

def call_google_vision_api(image_path):
    global emotion_prediction

    # Open the image and send it to Google Vision API
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform face detection and extract emotions
    response = client.face_detection(image=image)
    faces = response.face_annotations

    if faces:
        for face in faces:
            emotions = {
                'joy': face.joy_likelihood,
                'sorrow': face.sorrow_likelihood,
                'anger': face.anger_likelihood,
                'surprise': face.surprise_likelihood
            }
            # Sort emotions by likelihood (you can modify how you want to choose the dominant emotion)
            predicted_emotion = max(emotions, key=lambda k: emotions[k])
            emotion_prediction = {"emotion": predicted_emotion, "confidence": emotions[predicted_emotion]}
            print("Emotion Prediction:", emotion_prediction)
    else:
        print("No faces or emotions detected.")

    if response.error.message:
        raise Exception(f'{response.error.message}')

def main():
    global last_captured_time

    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)
    
    # Wait for the camera to initialize
    time.sleep(2)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image")
            break
        
        # Convert to grayscale (black and white)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face using OpenCV's built-in classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the detected faces and display the live feed
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display emotion prediction if available
            if emotion_prediction:
                predicted_emotion = emotion_prediction["emotion"]
                confidence = emotion_prediction["confidence"]
                text = f"{predicted_emotion}: {confidence}"

                # Add emotion text above the detected face
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the live camera feed with the face rectangle and emotion
        cv2.imshow('Live Camera', frame)

        # Check if 10 seconds have passed since the last capture
        current_time = time.time()
        if current_time - last_captured_time >= 10:
            capture_face(frame, gray, faces)
            last_captured_time = current_time

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
