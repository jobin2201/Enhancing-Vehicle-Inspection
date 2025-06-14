import os
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from PIL import Image
import numpy as np
import cv2
import streamlit as st
import mediapipe as mp

# Use MediaPipe solutions for face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Constants
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.6
LIP_DISTANCE_THRESHOLD = 5.0
JSON_FILE = "drowsiness_results.json"

# Create global variables to track drowsiness and yawning time
drowsy_time = 0
yawn_time = 0


def calculate_EAR(landmarks, eye_indices):
    A = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    B = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    C = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    EAR = (A + B) / (2.0 * C)
    return EAR


def calculate_MAR(landmarks, mouth_indices):
    A = np.linalg.norm(landmarks[mouth_indices[0]] -
                       landmarks[mouth_indices[2]])
    B = np.linalg.norm(landmarks[mouth_indices[1]] -
                       landmarks[mouth_indices[3]])
    C = np.linalg.norm(landmarks[mouth_indices[4]] -
                       landmarks[mouth_indices[5]])
    MAR = (A + B) / (2.0 * C)
    return MAR


def detect_eyes_and_yawning(image):
    global drowsy_time, yawn_time

    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return image, "No face detected", "N/A", "N/A", drowsy_time, yawn_time

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        landmarks = [(int(lm.x * w), int(lm.y * h))
                     for lm in face_landmarks.landmark]

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [263, 387, 385, 362, 380, 373]

        left_EAR = calculate_EAR(np.array(landmarks), left_eye_indices)
        right_EAR = calculate_EAR(np.array(landmarks), right_eye_indices)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        mouth_indices = [78, 308, 82, 312, 13, 14]
        MAR = calculate_MAR(np.array(landmarks), mouth_indices)
        lip_distance = np.linalg.norm(
            np.array(landmarks[13]) - np.array(landmarks[14]))

        # Eye status
        eye_status = "Awake" if avg_EAR > EAR_THRESHOLD else "Drowsy"

        # Mouth status logic
        mouth_status = "Not Yawning"
        if lip_distance < LIP_DISTANCE_THRESHOLD:
            mouth_status = "Not Yawning"
        elif MAR > MAR_THRESHOLD or lip_distance > LIP_DISTANCE_THRESHOLD:
            mouth_status = "Yawning"

        # Final status logic
        if eye_status == "Drowsy" and mouth_status == "Not Yawning":
            final_status = "Drowsy (Eye closed)"
        elif eye_status == "Awake" and mouth_status == "Yawning":
            final_status = "Drowsy (Yawning)"
        elif eye_status == "Awake" and mouth_status == "Not Yawning":
            final_status = "Awake"
        elif eye_status == "Drowsy" and mouth_status == "Yawning":
            final_status = "Drowsy (Eye closed and Yawning)"
        else:
            final_status = "Awake"

        # Update drowsy_time and yawn_time
        drowsy_time_val = 1 if final_status.startswith("Drowsy") else 0
        yawn_time_val = 1 if "Yawning" in final_status else 0

        # Draw landmarks and display text
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(
                                      color=(0, 255, 0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

        # Display results on image
        cv2.putText(image, f"EAR: {avg_EAR:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Eye Status: {eye_status}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if eye_status == "Drowsy" else (0, 255, 0), 2)
        cv2.putText(image, f"MAR: {MAR:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Lip Distance: {lip_distance:.2f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Mouth Status: {mouth_status}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if mouth_status == "Yawning" else (0, 255, 0), 2)
        cv2.putText(image, f"Final Status: {final_status}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if "Drowsy" in final_status else (0, 255, 0), 2)

        return image, eye_status, mouth_status, final_status, drowsy_time_val, yawn_time_val


def save_results(result):
    """Save results to a JSON file."""
    if os.path.isfile(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    existing_results.append(result)
    with open(JSON_FILE, "w") as f:
        json.dump(existing_results, f, indent=4)


def display_messages(final_status):
    messages = []
    if "Drowsy" in final_status and "Yawning" not in final_status:
        messages.append("**Message:** The driver is drowsy (eye closed).")
    if "Yawning" in final_status:
        messages.append("**Message:** The driver is yawning!")
    if not messages:
        messages.append("**Message:** Driver is alert.")

    for msg in messages:
        if "drowsy" in msg.lower() or "yawning" in msg.lower():
            st.warning(msg)
        else:
            st.info(msg)


def main_drowsiness_detection():
    st.title("🚗 Driver Drowsiness Detection")

    st.sidebar.title("Input Options")
    input_option = st.sidebar.radio(
        "Choose Input Type:", ("Image Upload", "Webcam"))

    if input_option == "Image Upload":
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image.convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            processed_image, eye_status, mouth_status, final_status, drowsy_time_val, yawn_time_val = detect_eyes_and_yawning(
                image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image, channels="RGB",
                     caption="Processed Image", use_column_width=True)

            result = {
                "Image Name": uploaded_file.name,
                "Eye Status": eye_status,
                "Mouth Status": mouth_status,
                "Final Status": final_status,
                "Drowsy Time": drowsy_time_val,
                "Yawning Time": yawn_time_val
            }

            display_messages(final_status)
            st.write("### Status Details:")
            st.write(f"**Eye Status:** {eye_status}")
            st.write(f"**Mouth Status:** {mouth_status}")
            st.write(f"**Final Status:** {final_status}")
            save_results(result)

    elif input_option == "Webcam":
        st.subheader("Webcam Stream")

        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                processed_img, eye_status, mouth_status, final_status, drowsy_time_val, yawn_time_val = detect_eyes_and_yawning(
                    img)
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                return av.VideoFrame.from_ndarray(processed_img, format="rgb24")

        webrtc_streamer(
            key="example", video_processor_factory=VideoProcessor, mode=WebRtcMode.SENDRECV)

        if st.button("Capture Frame"):
            st.session_state.capture = True  # Trigger the capture

        if st.session_state.get("capture"):
            st.session_state.capture = False
            # Logic to capture the frame and process it can be added here


# Check if this script is run directly
if __name__ == "__main__":
    main_drowsiness_detection()
