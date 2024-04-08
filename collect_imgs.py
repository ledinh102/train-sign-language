import os
import cv2
from labels import labels_dict

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 100

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Failed to open the camera.")
    exit(1)

try:
    for j in range(0, number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        print("Collecting data for class {}".format(labels_dict[j]))

        done = False
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            cv2.putText(
                frame,
                format('Ready to "' + labels_dict[j] + '"? Press "Q" ! :)'),
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) == ord("q"):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            cv2.imshow("frame", frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame)

            counter += 1

except KeyboardInterrupt:
    print("KeyboardInterrupt: Exiting program.")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
