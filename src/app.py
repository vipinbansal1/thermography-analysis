import streamlit as st
from PIL import Image, ImageDraw
from gradio_client import Client, handle_file
import shutil
import cv2
import math
import os
import tempfile
import time
import numpy as np
from videoanalytics import analysisVideoContent


default_prompt = """**Exhaustive Prompt (Balanced Thermography + Mixed Diameters):**
            *"Generate a high-resolution thermographic infrared image of an industrial setup with **many pipes of mixed diameters** (large, medium, and small), arranged in dense formation. Pipes should carry hot liquid and be shown in **balanced thermal color distribution** across surfaces and surroundings, not dominated by red. Use a vivid false-color thermography palette with **red, yellow, green, blue, and purple gradients**, clearly segmented to highlight different temperature ranges. Ensure multiple pipes show varying heat levels—some red, some yellow, some green, some blue—so the thermal map looks realistic and diverse. The style should resemble a professional thermal imaging scan with sharp contrasts, smooth gradients, and clear analytical color mapping.""",

generatedImagepath = "data/generated.png"
video_path = "data/video.mp4"

# Dummy image generation function
def generate_image(prompt) -> Image.Image:
    
    client = Client("bytedance-research/UNO-FLUX")
    result = client.predict(
            prompt=prompt,
            width=512,
            height=512,
            guidance=4,
            num_steps=25,
            seed=-1,
            image_prompt1=handle_file('data/4.jpg'),
            image_prompt2=handle_file('data/1.jpg'),
            image_prompt3=handle_file('data/3.jpg'),
            image_prompt4=handle_file('data/2.jpg'),
            api_name="/gradio_generate"
    )
    temp_image_path = result[0]  # first image path
    shutil.copy(temp_image_path, generatedImagepath)
    img = Image.open(generatedImagepath)
    return img
      

def simulate_heat_video(
    input_image_path,
    output_path,
    frame_width=640,
    frame_height=480,
    fps=30,
    duration_seconds=5,
    label_colors=None
):
    """
    Generate a simulated thermal video from a static image.

    Parameters:
        input_image_path (str): Path to input image.
        output_path (str): Path to save the output MP4 file.
        frame_width (int): Output video width.
        frame_height (int): Output video height.
        fps (int): Frames per second.
        duration_seconds (int): Total video duration.
        label_colors (dict): Optional mapping of label IDs to RGB colors.
    """

    # ==== DEFAULT COLORS ====
    if label_colors is None:
        label_colors = {
            0: [0, 0, 0],        # Background
            1: [255, 0, 0],      # Red (Hot)
            2: [255, 255, 0],    # Yellow
            3: [0, 255, 0],      # Green
            4: [0, 0, 255]       # Blue (Cold)
        }

    total_frames = fps * duration_seconds

    # ==== LOAD IMAGE ====
    base_image = Image.open(input_image_path).convert("RGB")
    base_image = base_image.resize((frame_width, frame_height))
    base_np = np.array(base_image).astype(np.float32)

    # ==== VIDEO WRITER ====
    # Create temporary file for video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

    # ==== FRAME SIMULATION ====
    for i in range(total_frames):
        t = i / total_frames
        frame = base_np.copy()

        # Heat wave oscillation
        heat_wave = (np.sin(2 * math.pi * t * 2) + 1) / 2

        # Adjust RGB values for simulation
        frame[..., 0] += 80 * heat_wave
        frame[..., 1] -= 40 * heat_wave
        frame[..., 2] -= 40 * (1 - heat_wave)

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # === SEGMENTATION ===
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        mask[(r > 200) & (g < 120) & (b < 120)] = 1  # Red
        mask[(r > 200) & (g > 200) & (b < 150)] = 2  # Yellow
        mask[(r < 150) & (g > 180) & (b < 150)] = 3  # Green
        mask[(r < 120) & (g < 120) & (b > 180)] = 4  # Blue

        # Overlay creation
        overlay = np.zeros_like(frame)
        for label, color in label_colors.items():
            overlay[mask == label] = color

        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # Calculate red area %
        red_pixels = np.sum(mask == 1)
        total_pixels = frame_width * frame_height
        red_percentage = (red_pixels / total_pixels) * 100

        cv2.putText(
            blended_bgr,
            f"Red Area: {red_percentage:.1f}%",
            (frame_width - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        video_writer.write(blended_bgr)

    video_writer.release()
    st.success(f"video generated successfully")
    size_in_bytes = os.path.getsize(temp_video_path)
    st.success(f"Video size: {size_in_bytes} bytes")
   
    shutil.copy(temp_video_path, video_path)
   
    with open(temp_video_path, "rb") as f:
        video_bytes = f.read()  # Convert to bytes
        video_stream=video_bytes
       
    st.success(f"video_stream completed  {len(video_stream)}")
    return video_stream


# Dummy action methods
def image_generator(prompt: str):
    with st.spinner("Processing... Please wait."):
        img = generate_image(prompt)
    return img

def video_generator():
    with st.spinner("Processing... Please wait."):
        video_stream = simulate_heat_video(
        input_image_path=generatedImagepath,
        output_path=video_path,
        frame_width=640,
        frame_height=480,
        fps=30,
        duration_seconds=2
        )
        time.sleep(3)  # simulate unknown long task
    return video_stream#"✅ Action 2 completed successfully!"

def video_analytics():
    st.success("[Video Analytics] Dummy action executed")

# -----------------------
# Session state
if "mode" not in st.session_state:
    st.session_state.mode = "input"
if "generated_img" not in st.session_state:
    st.session_state.generated_img = None
if "generated_video" not in st.session_state:
    st.session_state.generated_video = None

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>🔥 ThermoGraphy Analysis 📊</h1>", unsafe_allow_html=True)
# Layout with two columns (left and right)
left_col, right_col = st.columns([1, 5])  # left smaller, right bigger

with left_col:
    st.header("Menu")
    if st.button("Image Generation"):
        st.session_state.mode = "image"
    if st.button("Video Generation"):
        st.session_state.mode = "video"
    if st.button("Video Analytics"):
        st.session_state.mode = "analytics"

with right_col:
    st.header("Action")
    
    if st.session_state.mode == "input":
        user_prompt = st.text_area(
            "Type your thermography prompt here...",
            placeholder=default_prompt,
            height=300, width=600
        )
        if st.session_state.generated_img is None and st.session_state.mode=="image":
            st.session_state.generated_img = image_generator(user_prompt.strip() or default_prompt)

    elif st.session_state.mode == "image":
        if st.session_state.generated_img is None:
            st.session_state.generated_img = image_generator(default_prompt)
        st.image(st.session_state.generated_img, caption="Generated Thermography Image", use_container_width=False)

    elif st.session_state.mode == "video":
        if st.session_state.generated_video is None:
            st.session_state.generated_video = video_generator()
        st.video(st.session_state.generated_video)
    
    elif st.session_state.mode == "analytics":
        # Hide everything and show only analytics output
        st.session_state.data_frame = analysisVideoContent(st.session_state.generated_video, st.session_state.get("data_frame"))
        