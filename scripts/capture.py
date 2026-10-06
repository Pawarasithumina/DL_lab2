import cv2
import os
from datetime import datetime


# Class names and folders
classes = {
    "1": ("pawara", "friend_1"),
    "2": ("uchitha", "friend_2"),
    "3": ("hirusha", "friend_3"),
    "4": ("unknown", "unknown")
}


# Select class
print("Select the class:")
print("1 - Pawara")
print("2 - Uchitha")
print("3 - Hirusha")
print("4 - Unknown")

choice = input("Enter choice: ")

if choice not in classes:
    print("Invalid choice.")
    exit()

class_name, folder_name = classes[choice]

save_dir = os.path.join("raw", folder_name)

os.makedirs(save_dir, exist_ok=True)


# Load Haar Cascade
cascade_path = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade.")
    exit()


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# Count existing images
count = len([
    file for file in os.listdir(save_dir)
    if file.lower().endswith(".jpg")
])


print()
print("Class:", class_name)
print("Saving to:", save_dir)
print("Press SPACE to capture.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    # Display information
    cv2.putText(
        frame,
        f"Class: {class_name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Images: {count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show webcam
    cv2.imshow("Find My Friend Dataset", frame)


    # Read keyboard
    key = cv2.waitKey(1) & 0xFF


    # Capture image
    if key == ord(" "):

        if len(faces) == 0:

            print("No face detected. Image was not saved.")

        else:

            # Select largest detected face
            face = max(
                faces,
                key=lambda item: item[2] * item[3]
            )

            x, y, w, h = face


            # 20% padding
            padding_x = int(w * 0.20)
            padding_y = int(h * 0.20)

            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)

            x2 = min(
                frame.shape[1],
                x + w + padding_x
            )

            y2 = min(
                frame.shape[0],
                y + h + padding_y
            )


            # Crop face
            face_crop = frame[y1:y2, x1:x2]


            # Create timestamp
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )


            # Create filename
            filename = (
                f"{class_name}_{timestamp}_{count + 1}.jpg"
            )

            filepath = os.path.join(
                save_dir,
                filename
            )


            # Save image
            cv2.imwrite(filepath, face_crop)

            count += 1

            print("Saved:", filepath)


    # Quit
    elif key == ord("q"):

        break


# Release resources
cap.release()
cv2.destroyAllWindows()

print()
print("Finished.")
print("Class:", class_name)
print("Total images:", count)
