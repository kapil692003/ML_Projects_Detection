import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera or video file
cap = cv2.VideoCapture("Interns.mp4")
frame_count = 0
nth_frame = 3 

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    # If the frame was not grabbed, break the loop
    if not ret:
        print("Error: Frame not read properly or end of video.")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (480, 900))  # Adjust dimensions as necessary

    # Process every nth frame
    if frame_count % nth_frame == 0:
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Increment the frame count
    frame_count += 1

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
