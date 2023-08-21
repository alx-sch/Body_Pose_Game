import cv2
import time
import mediapipe as mp
import os
import numpy as np
import sys
import random
from tensorflow.keras.models import load_model
from utils_2 import (
    initialize_pose_model,
    process_frame,
    draw_bold_text,
    extract_landmarks,
    predict_pose,
)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

label_mapping = {"X": 0, "crane": 1, "empty": 2, "squat": 3, "stand": 4}
POSES = ["stand", "squat", "X", "empty"]

COUNTDOWN = int(sys.argv[1])
ROUNDS = int(sys.argv[2])

rect_top_left = (950, 20)
rect_bottom_right = (350, 700)
rect_color = (0, 255, 0)

model = load_model("./models/model_TH_1.h5")


def main():
    cap = cv2.VideoCapture(0)

    last_capture_time = time.time()
    countdown = False
    points = 0
    count = 0  # Initialize count to 0
    rounds_left = ROUNDS

    pose = initialize_pose_model()

    while True:
        success, frame = cap.read()

        if success:
            cv2.rectangle(
                frame, rect_top_left, rect_bottom_right, rect_color, thickness=4
            )

            results, frame = process_frame(frame, pose)
            frame = cv2.flip(frame, 1)

            if countdown:
                if rounds_left > 0:
                    if count == 0:
                        count = COUNTDOWN
                        current_pose = random.choice(POSES)

                    text = f"{current_pose.upper()}: {count} s"
                    text_position = (30, 100)

                    draw_bold_text(frame, text, text_position, color=(0, 0, 255))

                    """
                    # display frame / videostream in window
                    cv2.imshow("Pose Detection", frame)
                    """

                    if time.time() - last_capture_time >= 1:
                        count -= 1
                        last_capture_time = time.time()

                    if count == 0:
                        timestamp = int(time.time())

                        landmark_coordinates = extract_landmarks(results)

                        ###
                        landmarks_folder = f"test_play/landmarks"
                        picture_folder = f"test_play/pics"

                        if not os.path.exists(picture_folder):
                            os.makedirs(picture_folder)
                        if not os.path.exists(landmarks_folder):
                            os.makedirs(landmarks_folder)

                        filename_pictures = f"{picture_folder}/{timestamp}.jpg"
                        filename_landmarks = f"{landmarks_folder}/{timestamp}.npy"
                        ####

                        cv2.imwrite(filename_pictures, frame)
                        np.save(filename_landmarks, landmark_coordinates)

                        predicted_pose = predict_pose(
                            landmark_coordinates, label_mapping, model
                        )

                        print(predicted_pose)
                        print(f"rounds left: {rounds_left}")

                        rounds_left -= 1

            else:
                text = "Press SPACE to start the game"
                text_position = (30, 100)

                draw_bold_text(
                    frame,
                    text,
                    text_position,
                    font_scale=2,
                    color=(255, 0, 0),
                    thickness=2,
                    offset=1,
                )

            # display frame / videostream in window
            cv2.imshow("Pose Detection", frame)

            key = cv2.waitKey(5)
            if key == ord("q"):  # Press 'q' to exit
                break

            if key == ord(" "):
                countdown = (True,)
                last_capture_time = time.time()

        else:
            print("Error: Could not open camera.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
