import cv2
import streamlit as st
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta

from pixel_count_finder import detect_green_zones, detect_red_zones, detect_yellow_zones
from boundingbox import bounding_box_bet_frames
from color_analysis import get_color_densities
import google.generativeai as genai
color_data =[]
def analysisVideoContent(video_stream, data):

    genai.configure(api_key="<Pass the key>")
    model = genai.GenerativeModel("gemini-1.5-flash")

    def ask_gemini(prompt, df=None):
       
        if df is not None:
            content = prompt + "\n\nHere is the thermal data:\n" + df.to_csv(index=False)
        else:
            content = prompt
        try:
            response = model.generate_content(content)
            return response.text
        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"

    csv_path = "output/color_intensity_Data.csv"

    def analytics(df):
        with st.expander("🌡️ AI-Powered Thermal Analysis Summary", expanded=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📌 Summary Insights")
                summary_prompt = (
                    "You are a thermal imaging analyst. Analyze this dataset to identify:\n"
                    "- Total frames analyzed\n"
                    "- Frames with abnormal temperature (deep red intensity)\n"
                    "- Time ranges where high intensity occurred\n"
                    "- Any visible spikes or concerning patterns\n"
                    "- Write in markdown format with bold titles and bullet points"
                )
                with st.spinner("Gemini is analyzing thermal patterns..."):
                    summary = ask_gemini(summary_prompt, df)
                st.markdown(summary)

            with col2:
                st.subheader("💬 Ask ThermalBot")
                user_query = st.text_area("Ask any question about the thermal analysis:")
                if st.button("Ask Me"):
                    with st.spinner("Analyzing your query..."):
                        response = ask_gemini(user_query, df)
                    st.info(response)
   
    st.title("🎥 Real-Time Temperature Intensity Monitoring with Alerts")

    # Resize helper
    def resize_frame(frame, height=240):
        h, w = frame.shape[:2]
        scale = height / h
        return cv2.resize(frame, (int(w * scale), height))

    # Detect color jumps
    def detect_color_jump(curr, prev, threshold=400):
        alert_colors = []
        if prev:
            for color in curr:
                if abs(curr[color] - prev[color]) > threshold:
                    alert_colors.append(color)
        return alert_colors
    
    
    df = data
    if True or start:
        if True or video_file:
            
            if data is None or data.empty:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                #tfile.write(video_file.read())
                tfile.write(video_stream)

                cap1 = cv2.VideoCapture(tfile.name)
                cap2 = cv2.VideoCapture(tfile.name)
                cap3 = cv2.VideoCapture(tfile.name)
                cap4 = cv2.VideoCapture(tfile.name)

                col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                placeholder1 = col1.empty()
                placeholder2 = col2.empty()
                placeholder3 = col3.empty()
                placeholder4 = col4.empty()

                plot_placeholder_left = col1.empty()
                plot_placeholder_right = col3.empty()
                plot_placeholder_color = col4.empty()
                alert_placeholder = col4.empty()

                frame_count = 0
                x_vals_left, y_vals_left = [], []
                x_vals_right, y_vals_right = [], []
                x_vals_color, red_vals, yellow_vals, green_vals, blue_vals = [], [], [], [], []

                prev_frame1, prev_frame3 = None, None
                prev_intensity = None

                fps = cap1.get(cv2.CAP_PROP_FPS) or 30
    
                while True:
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    ret3, frame3 = cap3.read()
                    ret4, frame4 = cap4.read()

                    if not (ret1 and ret2 and ret3 and ret4):
                        break

                    frame1, dead_pixel_for_frame1 = detect_red_zones(frame1)
                    frame3, dead_pixel_for_frame3 = detect_yellow_zones(frame3)

                    if isinstance(prev_frame1, np.ndarray):
                        temp = bounding_box_bet_frames(prev_frame1, frame1)
                        if isinstance(temp, np.ndarray):
                            frame1 = temp
                    if isinstance(prev_frame3, np.ndarray):
                        temp = bounding_box_bet_frames(prev_frame3, frame3)
                        if isinstance(temp, np.ndarray):
                            frame3 = temp
                    prev_frame1 = frame1
                    prev_frame3 = frame3

                    frame_count += 1
                    timestamp = timedelta(seconds=frame_count / fps)
                    timestamp_str = str(timestamp).split(".")[0] + "." + str(timestamp.microseconds // 1000).zfill(3)

                    x_vals_left.append(frame_count)
                    y_vals_left.append(dead_pixel_for_frame1)
                    x_vals_right.append(frame_count)
                    y_vals_right.append(dead_pixel_for_frame3)

                    r, y, g, b = get_color_densities(frame4)
                    x_vals_color.append(frame_count)
                    red_vals.append(r)
                    yellow_vals.append(y)
                    green_vals.append(g)
                    blue_vals.append(b)

                    curr_intensity = {"red": r, "yellow": y, "green": g, "blue": b}
                    color_data.append({
                        "Frame": frame_count,
                        "Timestamp": timestamp_str,
                        **curr_intensity
                    })

                    jump_alerts = detect_color_jump(curr_intensity, prev_intensity)
                    prev_intensity = curr_intensity

                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
                    frame4_disp = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)

                    f1_resized = resize_frame(frame1)
                    f2_live = resize_frame(frame2, height=480)
                    f3_resized = resize_frame(frame3)
                    f4_resized = resize_frame(frame4_disp)

                    placeholder1.image(f1_resized, caption="Red Zones")
                    placeholder2.image(f2_live, caption="Live View")
                    placeholder3.image(f3_resized, caption="Yellow Zones")

                    # Display alert-causing frame in Window 4
                    if jump_alerts:
                        alert_msg = (
                            f"🚨 <b>ALERT</b>: Sudden spike in {', '.join(jump_alerts).upper()} "
                            f"(Frame {frame_count}, Time {timestamp_str})"
                        )
                        placeholder4.image(f4_resized, caption="ALERT Frame")
                        alert_placeholder.markdown(
                            f"<div style='color:red; font-weight:bold;'>{alert_msg}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        alert_placeholder.markdown("")

                    # Graphs
                    fig_left, ax_left = plt.subplots(figsize=(4, 2.5))
                    ax_left.plot(x_vals_left, y_vals_left, color='red')
                    ax_left.set_title("High Temperature Distribution")
                    ax_left.set_xlabel("Frames ->")
                    ax_left.set_ylabel("Intensity")
                    ax_left.grid(True)
                    plot_placeholder_left.pyplot(fig_left)
                    plt.close(fig_left)

                    fig_right, ax_right = plt.subplots(figsize=(4, 2.5))
                    ax_right.plot(x_vals_right, y_vals_right, color='gold')
                    ax_right.set_title("Moderate Temperature Distribution")
                    ax_right.set_xlabel("Framees ->")
                    ax_right.set_ylabel("Intensity")
                    ax_right.grid(True)
                    plot_placeholder_right.pyplot(fig_right)
                    plt.close(fig_right)

                    fig_color, ax_color = plt.subplots(figsize=(4, 3))
                    ax_color.plot(x_vals_color, red_vals, color='red', label='Red')
                    ax_color.plot(x_vals_color, yellow_vals, color='yellow', label='Yellow')
                    ax_color.plot(x_vals_color, green_vals, color='green', label='Green')
                    ax_color.plot(x_vals_color, blue_vals, color='blue', label='Blue')
                    ax_color.set_title("Temperature Distribution")
                    ax_color.set_xlabel("Frames - >")
                    ax_color.set_ylabel("Normalized Intensity")
                    ax_color.legend()
                    ax_color.grid(True)
                    plot_placeholder_color.pyplot(fig_color)
                    plt.close(fig_color)

                    # Faster loop
                    time.sleep(1 / 240)

                os.makedirs("output", exist_ok=True)
                out_path = "output/color_intensity_Data.csv"
                pd.DataFrame(color_data).to_csv(out_path, index=False)

                cap1.release()
                cap2.release()
                cap3.release()
                cap4.release()
                df = pd.DataFrame(color_data)

                # Convert DataFrame to CSV string
                csv_string_pandas = df.to_csv(index=False)


                analytics(df)
                #return color_data
            else:
                analytics(data)
        else:
            st.warning("Please upload a video file first.")
    else:
        st.info("Upload a video and click 'Start Playback'.")
    return df

    # ===============================
    # ✅ AI-Powered Thermal Summary and Q&A
    # ===============================

   
   
