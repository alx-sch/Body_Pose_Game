import cv2
import time
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

POSES = ["stand", "squat", "X"]


def process_frame(frame, pose):
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    return frame


def main():
    cap = cv2.VideoCapture(0)  # Open the camera

    last_capture_time = time.time()
    countdown = False
    count = 5
    current_pose_index = 0

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
    ) as pose:
        while True:
            success, frame = cap.read()

            if success:
                frame = process_frame(frame, pose)

                if countdown:
                    cv2.putText(
                        frame,
                        f"Taking picture in for '{POSES[current_pose_index]}': {count}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if time.time() - last_capture_time >= 1:
                        count -= 1
                        last_capture_time = time.time()

                    if count == 0:
                        pose_folder = POSES[current_pose_index]
                        if not os.path.exists(pose_folder):
                            os.makedirs(pose_folder)
                        timestamp = int(time.time())
                        filename = f"{pose_folder}/picture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)

                        count = 5
                        countdown = False
                        current_pose_index = (current_pose_index + 1) % len(POSES)
                else:
                    cv2.putText(
                        frame,
                        "Press SPACE to start countdown",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Pose Detection", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):  # Press 'q' to exit
                    break

                if key == ord(" "):
                    countdown = True
                    last_capture_time = time.time()

            else:
                print("Error: Could not open camera.")
                break


if __name__ == "__main__":
    main()
