import csv

# import the modules we developed
from ML_Bayesian import get_next_parameters
from image_process import analyze_image
from camera_capture_fake import capture_image



HISTORY_FILE = "experiment_history.csv"
def get_next_generation(history_file):
    with open(history_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        return 1

    last_generation = int(rows[-1]["generation"])
    return last_generation + 1
generation = get_next_generation(HISTORY_FILE)

# Step 1: Get next mixing parameters from BO
next_params = get_next_parameters(HISTORY_FILE)

mix_ratio = next_params["mix_ratio"]
mix_time = next_params["mix_time"]

print("\n===== Suggested Mixing Parameters =====")
print("mix_ratio =", mix_ratio)
print("mix_time =", mix_time)

# Step 2: Mixing workflow
# Here the mixing workflow will run.
# The mixing parameters (mix_ratio, mix_time) should be passed to the liquid handling system.

# Step 3: Robotic transfer workflow
# After mixing, the robotic arm transfers the material
# from the liquid handling system to the printer.

# Step 4: Printing step
# The printer prints the pattern using the prepared material.

# Step 5: Capture image of printed result
image_path = capture_image()

# Step 6: Image processing
result = analyze_image(image_path)

score = result["score"]

print("Score =", score)

# Step 7: Record experiment data
# Append this generation result to the history file
with open(HISTORY_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        generation,
        mix_ratio,
        mix_time,
        score
    ])

print("New experiment saved to", HISTORY_FILE)
print("generation =", generation)
print("mix_ratio =", mix_ratio)
print("mix_time =", mix_time)
print("score =", score)
