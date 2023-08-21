import cv2
import time
import mediapipe as mp
import os
import numpy as np
from utils_2 import (
    initialize_pose_model,
    process_frame,
    draw_bold_text,
    extract_landmarks,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

POSES = ["other"]
# POSES = ["stand", "squat", "X", "crane", "empty"]
POSES = [POSES[-1]] + POSES[:-1]

COUNTDOWN = 2

ITERATIONS_PER_POSE = 20
ITERATIONS_PER_POSE = ITERATIONS_PER_POSE * len(POSES)

data_folder = "data_collecting"

rect_top_left = (950, 20)
rect_bottom_right = (350, 700)
rect_color = (0, 255, 0)


def main():
    cap = cv2.VideoCapture(0)  # Open the camera

    last_capture_time = time.time()
    countdown = False
    count = 0  # Initialize count to 0
    iterations_left = ITERATIONS_PER_POSE
    current_pose_index = 0

    pose = initialize_pose_model()

    while True:
        success, frame = cap.read()

        if success:
            cv2.rectangle(
                frame, rect_top_left, rect_bottom_right, rect_color, thickness=4
            )

        if success:
            results, frame = process_frame(frame, pose)

            # mirror frame
            frame = cv2.flip(frame, 1)

            if countdown:
                if iterations_left > 0:
                    if (
                        count == 0
                    ):  # Initialize count to 5 when starting a new pose iteration
                        count = COUNTDOWN
                        current_pose_index = (current_pose_index + 1) % len(
                            POSES
                        )  # Move to the next pose

                    current_pose = POSES[current_pose_index]
                    round_no = int(np.ceil(iterations_left / len(POSES)))

                    # Display pose & countdown on frame:
                    text = f"{current_pose.upper()} (left: {round_no}): {count} s"
                    text_position = (30, 100)

                    draw_bold_text(frame, text, text_position, color=(0, 0, 255))

                    # Display the frame
                    cv2.imshow("Pose Detection", frame)

                    if time.time() - last_capture_time >= 1:
                        count -= 1
                        last_capture_time = time.time()

                    if count == 0:
                        picture_folder = f"{data_folder}/pictures/{current_pose}"
                        landmarks_folder = f"{data_folder}/landmarks/{current_pose}"
                        if not os.path.exists(picture_folder):
                            os.makedirs(picture_folder)
                        if not os.path.exists(landmarks_folder):
                            os.makedirs(landmarks_folder)
                        timestamp = int(time.time())

                        landmark_coordinates = extract_landmarks(results)

                        filename_pictures = (
                            f"{picture_folder}/{current_pose}_{timestamp}.jpg"
                        )
                        filename_landmarks = (
                            f"{landmarks_folder}/{current_pose}_{timestamp}.npy"
                        )

                        cv2.imwrite(filename_pictures, frame)
                        np.save(filename_landmarks, landmark_coordinates)

                        iterations_left -= 1

                else:
                    countdown = False
                    iterations_left = ITERATIONS_PER_POSE

            else:
                text = "Press SPACE to start countdown"
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
                countdown = True
                last_capture_time = time.time()

        else:
            print("Error: Could not open camera.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
