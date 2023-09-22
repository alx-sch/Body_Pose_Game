import streamlit as st
import subprocess

from utils_2 import (
    initialize_pose_model,
    process_frame,
    blink_screen,
    draw_bold_text,
    extract_landmarks,
    save_data,
    predict_pose_v2,
    display_gameover_message,
)


def main():
    st.title("Strike A Pose Game")

    # Add Streamlit widgets here to take user inputs if needed
    # Embed the webcam widget using an iframe.
    st.components.v1.iframe("pages/webcam.html", width=640, height=480)

    # Run the 'play.py' script as a subprocess
    if st.button("Start Game"):
        run_play_script()


def run_play_script():
    # Run the 'play.py' script using subprocess
    process = subprocess.Popen(
        ["python", "play.py", "10", "5"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for the subprocess to finish and capture its output
    stdout, stderr = process.communicate()

    # Display the output in Streamlit
    st.text("Game Output:")
    st.text(stdout.decode())
    st.text("Errors:")
    st.text(stderr.decode())


if __name__ == "__main__":
    main()
