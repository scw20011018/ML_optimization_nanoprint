README – Autonomous Nanoprinting Optimization Pipeline

🔷 Overview

This project implements an automated closed-loop workflow for optimizing nanoprinting parameters using image feedback and machine learning.

The system simulates and integrates:

Parameter prediction (Bayesian Optimization)

Mixing process (simulated)

Image acquisition (fake camera or real camera)

Image processing for feature extraction

Data recording for iterative learning

The goal is to improve printing resolution and uniformity through continuous feedback.

🔷 Workflow Summary

The full pipeline follows this loop:
ML Prediction → Mixing → Printing → Image Capture → Image Processing → Score → Update History → Repeat

🔷 File Structure
main.py                 # Main workflow controller
ML_Bayesian.py         # Bayesian Optimization module
image_process.py       # Image processing and feature extraction
camera_capture.py      # Image capture (fake or real camera)
experiment_history.csv # Stores all experiment data


🔷 Step-by-Step Workflow

1️⃣ Parameter Prediction (ML)
next_params = get_next_parameters(HISTORY_FILE)

#Uses Bayesian Optimization
#Inputs: past experiment data
#Outputs:
mix_ratio
mix_time


2️⃣ Mixing (Simulated)
Mixing process (currently simulated)

#In real system:
#Executed by liquid handling system (Opentrons)
#In current version:
#Placeholder step


3️⃣ Printing
Printing step (assumed completed)

#Material is printed after mixing
#Output will be evaluated via imaging


4️⃣ Image Capture
Option A: Fake Camera (for testing)
capture_image("current_capture.jpg")

#Copies a sample image
#Simulates real experiment output
#Used before hardware integration

Option B: Real Camera (future integration)
IDS camera (e.g., U3-34E0XCP-C-HQ)
Controlled via Python SDK
Captures real printed structure


5️⃣ Image Processing
result = analyze_image(image_path)

#Key Steps:
1. Convert to grayscale
2. Threshold to binary image
3. Extract printing region (contour detection)
4. Determine structure orientation (horizontal / vertical)
5. Apply scan lines perpendicular to structure
6. Measure local widths using longest continuous white segment

📏 Resolution Calculation

#Measure width at multiple scan lines → width_list

Use:
resolution_metric = np.percentile(width_array, 10)

📊 Additional Metrics
mean_width
std_width
uniformity_index = std / mean
Mean → overall size
Std → variation
Uniformity → consistence

6️⃣ Score Calculation
score = resolution_um * (1 + uniformity_index)

#Lower score = better performance
(you can customize objective)

7️⃣ Record Data
writer.writerow([
    generation,
    mix_ratio,
    mix_time,
    score
])

#Saved into:
experiment_history.csv
This file is used for:
Tracking experiment results
Training the ML model



🔷 Experiment_history.csv Format
generation, mix_ratio, mix_time, score
1, ..., ..., ...
2, ..., ..., ...


🔷 Key Design Ideas
✔ Closed-loop optimization

Each experiment improves the next one.

✔ Image-based evaluation

No manual measurement needed.

✔ Modular structure

Each part can be replaced independently:

Fake camera → real camera

Simulated mixing → real system

🔷 How to Run
1 Install dependencies
pip install numpy opencv-python bayesian-optimization

2 Run workflow
python main.py

