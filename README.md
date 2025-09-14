🔥 Thermography Analysis with Synthetic Data & LLM Reporting

📖 Overview
This project presents a synthetic-data-driven framework for industrial pipe thermography monitoring.
It integrates synthetic image generation, temporal heat simulation, pixel-level anomaly detection, and LLM-powered reporting to enable predictive maintenance in data-scarce environments.

⚡ Features
1. Synthetic Thermography: Generate thermographic images of industrial pipes using UNO-Flux.
2. Temporal Simulation: Create video sequences with smooth pixel color transitions to simulate heat diffusion.
3. Frame-by-Frame Analysis: Extract thermal color bands (red, yellow, green, blue).
4. Anomaly Localization: Detect significant inter-frame variations and highlight with bounding boxes.
5. LLM Reporting: Generate structured human-readable reports with risk levels and recommended actions.

🛠️ Tech Stack
**Python:** OpenCV, NumPy, Matplotlib
**UNO-Flux:** For synthetic thermography image generation
**Google LLM:** For automated report generation

🚀 Setup & Installation
# Clone the repository
git clone https://github.com/vipinbansal1/thermography-analysis.git
cd thermography-analysis
# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
# Install dependencies
pip install -r requirements.txt

📂 Project Structure
thermography-analysis/
│-- data/                       # Synthetic images & videos
│-- src/
│   │-- app.py                  # Main streamlit app, that includes the image and video generation module
│   │-- videoanalytics.py       # Sub streamlit app for performing video analytics
│   │-- pixel_count_finder.py   # Pixel Count util file.
│   │-- boundignbox.py          # Bounding box detection
│   |-- color-analysis.py       # Util method to count the color pixels
│-- requirements.txt
│-- README.md

📊 Example Workflow
1. Generate synthetic thermography images using UNO-Flux.
2. Simulate temporal heat diffusion as a video.
3. Run analysis to extract thermal color bands & detect anomalies.
4. Send pixel statistics to an LLM to auto-generate a safety report.

📑 Sample Output
Anomaly detection bounding boxes on thermal frames.

📌 Future Work
Integration with real infrared thermography datasets.
Advanced temporal analysis using ConvLSTM / Transformers.
Deployment as an edge-AI solution for real-time industrial monitoring.

📜 License
MIT License – feel free to use and adapt this work with attribution.

