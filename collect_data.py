"""
COLLECTING TRAINING DATA
This script records and saves extracted body pose landmark coordinates and annotated frames
from a videostream at fixed time intervals.

Adjust parameters to modify number/name of poses and 
recording settings (no. of iterations, time between recordings, output directory).

Press 'Space' to initiate countdown and data collection (initially and after each round).
Press 'R' to restart the script execution.
Press 'Q' to quit the script.
"""

##############
### IMPORT ###
##############

import cv2  # OpenCV for video capture and processin
import time  # Time-related functions
import numpy as np  # Numerical operations

# Import custom utility functions
from utils import (
    initialize_pose_model,
    process_frame,
    draw_bold_text,
    save_data,
    display_instructions,
)

##################
### PARAMETERS ###
##################

# Name of poses that are recorded.
POSES = ["stand", "squat", "X", "empty"]

# Time between recordings in seconds.
COUNTDOWN = 5

# Number of recordings per pose.
RECORDINGS_PER_POSE = 10

# Folder / data path in which to save recorded data (extraxcted landmarks and annotated frames).
SAVE_PATH = "training_data"

# Settings for the "postion rectangle" (outlining the 'region of interest' (ROI) used for collecting data).
# The same ROI is applied in the game / pose detection.
RECT_TOP_LEFT = (950, 20)
RECT_BOTTOM_RIGHT = (350, 700)
RECT_COLOR = (0, 255, 0)  # green
RECT_THICKNESS = 4


#################
### EXECUTION ###
#################


def main():
    # Initializes recording variables.
    current_pose_index = 0
    countdown = False
    count = COUNTDOWN
    ITERATIONS_TOTAL = RECORDINGS_PER_POSE * len(POSES)
    iterations_left = ITERATIONS_TOTAL

    capture = cv2.VideoCapture(0)  # Opens the camera.

    # Initializes MediaPipe's pose landmark detection.
    pose = initialize_pose_model()

    while True:
        success, frame = capture.read()

        if success:
            # Draws ROI rectangle into the videostream.
            cv2.rectangle(
                frame,
                RECT_TOP_LEFT,
                RECT_BOTTOM_RIGHT,
                RECT_COLOR,
                thickness=RECT_THICKNESS,
            )

            # Processes videostream: Detects pose landmarks and draws them onto frame.
            results, frame = process_frame(frame, pose)

            # Mirrors the videostream, it looks uncanny otherwise ;)
            frame = cv2.flip(frame, 1)

            # Initiates countdown/data collection when 'space' is pressed.
            if countdown:
                if iterations_left > 0:  # Checks if data collection is ongoing.
                    # Pose index is updated when countdown is finished and data was collected (count = 0).
                    current_pose = POSES[current_pose_index]

                    # Number of data collection iterations left for each pose.
                    round_number = int(np.ceil(iterations_left / len(POSES)))

                    # Displays pose to be recorded and countdown until recording.
                    text = f"{current_pose.upper()} (left: {round_number}): {count} s"

                    draw_bold_text(frame, text, position=(30, 100), color=(0, 0, 255))

                    # Displays instructions on how to quit data collection.
                    text = "Press Q to quit"

                    draw_bold_text(
                        frame,
                        text,
                        position=(30, 170),
                        font_scale=1.5,
                        color=(0, 255, 255),
                        thickness=1,
                        offset=1,
                    )

                    # Updates countdown variable after one second has elapsed.
                    if time.time() - last_time_check >= 1:
                        count -= 1
                        last_time_check = time.time()

                    # Saves landmark coordinates and annotated frames when countdown reached zero.
                    if count == 0:
                        # Saves annotated frame and landmark coordinates in specified path.
                        save_data(SAVE_PATH, current_pose, frame, results)
                        print(
                            f"Data for '{current_pose.upper()}' ({RECORDINGS_PER_POSE - round_number + 1}/{RECORDINGS_PER_POSE}) saved in folder '{SAVE_PATH}'."
                        )

                        count = COUNTDOWN  # Resets countdown variable.
                        # Moves index to the next pose (looping through pose list).
                        current_pose_index = (current_pose_index + 1) % len(POSES)

                        iterations_left -= 1

                # Resets countdown parameters when data collection is completed (iterations_left = 0).
                else:
                    countdown = False
                    iterations_left = ITERATIONS_TOTAL

            # Displays instructions when data collection is not ongoing (countdown = False).
            else:
                display_instructions(frame)

            # Displays videostream in window.
            cv2.imshow("Collecting Body Pose Data", frame)

            key = cv2.waitKey(5)  # Records a key press every 5 ms.

            # Checks if 'q' is pressed and if so, exits the execition loop and terminates the application.
            if key == ord("q"):
                break

            # Checks if spacebar is pressed and if so, initializes countdown.
            if key == ord(" "):
                countdown = True
                last_time_check = time.time()

            # Checks if 'r' is pressed and if so, calls the 'main()' function to restart  the execution.
            if key == ord("r"):
                capture.release()
                cv2.destroyAllWindows()
                main()

        else:
            print("Error: Could not open camera.")
            break

    capture.release()
    cv2.destroyAllWindows()  # Close window used for displaying video.


if __name__ == "__main__":
    main()
