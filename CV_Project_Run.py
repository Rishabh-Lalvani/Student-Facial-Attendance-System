
import cv2
import os
import numpy as np
import pickle
from sklearn import neighbors
import pandas as pd
from datetime import datetime

IMAGE_FOLDER = "C:/Users/Rishabh/Desktop/Goofy Ahh python/Students"  
EXCEL_FILE = "C:/Users/Rishabh/Desktop/Goofy Ahh python/Test Run 1.xlsx"

# Load images and create training data
def load_images_from_folder(folder):
    images = []
    labels = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(folder, file_name)
            image = cv2.imread(image_path)
            if image is not None:
                # Convert image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                labels.append(file_name.split('.')[0])  # Using filename without extension as label
    return images, labels

# Encode faces using OpenCV
# Encode faces using OpenCV
def encode_faces(images, labels):
    face_encodings = []
    valid_labels = []
    
    for img, label in zip(images, labels):
        # Detect face using OpenCV's Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))  # Resize to a fixed size
            face_encodings.append(face.flatten())  # Flatten the image to a 1D array
            valid_labels.append(label)  # Add the corresponding label only if face is detected
            
    return face_encodings, valid_labels

def save_attendance_to_excel(attendance_records, excel_file):
    # Check if the Excel file exists
    if os.path.exists(excel_file):
        # Load existing data
        existing_df = pd.read_excel(excel_file)
        # Append new records to existing data
        new_df = pd.DataFrame(attendance_records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Create a new DataFrame if the file doesn't exist
        combined_df = pd.DataFrame(attendance_records)

    # Save to Excel
    combined_df.to_excel(excel_file, index=False)


# Main function
def main():
    # Load images and labels
    images, labels = load_images_from_folder(IMAGE_FOLDER)

    # Encode faces
    known_face_encodings, valid_labels = encode_faces(images, labels)

    # Ensure that we have the same number of encodings and labels
    if len(known_face_encodings) == 0:
        print("No faces found in the images. Please check your image folder.")
        return

    if len(known_face_encodings) != len(valid_labels):
        print(f"Mismatch in encodings and labels: {len(known_face_encodings)} encodings, {len(valid_labels)} labels.")
        return

    # Train KNN classifier
    n_neighbors = 3  # You can experiment with this value
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
    knn_clf.fit(known_face_encodings, valid_labels)

    # Save the trained KNN classifier
    with open('knn_face_classifier.pkl', 'wb') as f:
        pickle.dump(knn_clf, f)

    # Set up Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Set up webcam for attendance
    video_capture = cv2.VideoCapture(0)
    
    attendance_records = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Extract face and encode it
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100)).flatten().reshape(1, -1)
            
            # Predict the label (student's name)
            name = knn_clf.predict(face_resized)
            
            # Mark attendance
            attendance_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance_records.append({'Name': name[0], 'Time': attendance_time})

            # Draw a rectangle around the face and put the label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save attendance records to Excel
    save_attendance_to_excel(attendance_records, EXCEL_FILE)

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()