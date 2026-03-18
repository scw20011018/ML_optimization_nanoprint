# install the opencv and numpy package
import cv2
import numpy as np

def analyze_image(image_path):

    #Step 1: load the image
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Cannot load image: {image_path}")

    #Step 2: convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Step 3: Blur the image to reduce noise
    #blur = cv2.GaussianBlur(gray, (5,5), 0)

    #We only need to extract the features in a certain area of our print out material
    #Step 4: Transform the image to a binary image detection
    #Convert gray image into black and white image, the white region is the printed region
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    #Step 5: Remove white noise (clean up step)
    #Fill the holes in the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #Step 6: Find contours in the binary image
    #contours are the edges of the printed reigion
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No printed region detected.")
        return {"score": 0}

    #Step 7: Find all the contours
    min_area = 50
    #fill out small contours, which are less than 50 pixels
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            valid_contours.append(cnt)
    
    if len(valid_contours) == 0:
        print("No valid contours found.")
        return {"score": 0}
    
    main_contour = max(valid_contours, key=cv2.contourArea)

    #Step 8: form a mask from the main contour to extract the printed region
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)

    #Step 9: Calculate the continuity from the mask
    #Define the number ofindependent areas using connected components
    #If the the number is 1, it means the printed region is continuous
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #Reduce the background
    num_regions = num_labels - 1

    #Show the number of breakpoints
    breakpoints = max(0, num_regions - 1)

    # Step 10: Decide the main direction of the printed structure
    x, y, w, h = cv2.boundingRect(main_contour)

    width_list = []
    scan_positions = []
    scan_direction = ""

    # Helper function: find the longest continuous white segment
    def longest_white_run(line):
        max_run = 0
        current_run = 0

        for pixel in line:
            if pixel > 0:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 0

        return max_run

    # If the structure is taller than wide, it is mainly vertical
    # Use horizontal scan lines
    if h >= w:
        scan_direction = "horizontal"

        num_scan_lines = 20
        y_positions = np.linspace(y, y + h - 1, num_scan_lines, dtype=int)

        for y_pos in y_positions:
            row = mask[y_pos, x:x + w]
            local_width = longest_white_run(row)

            if local_width > 0:
                width_list.append(local_width)
                scan_positions.append(y_pos)

    # If the structure is wider than tall, it is mainly horizontal
    # Use vertical scan lines
    else:
        scan_direction = "vertical"

        num_scan_lines = 20
        x_positions = np.linspace(x, x + w - 1, num_scan_lines, dtype=int)

        for x_pos in x_positions:
            col = mask[y:y + h, x_pos]
            local_width = longest_white_run(col)

            if local_width > 0:
                width_list.append(local_width)
                scan_positions.append(x_pos)

    # Step 11: Calculate width statistics
    if len(width_list) == 0:
        print("No valid width measurement found.")
        return {"score": 0}

    width_array = np.array(width_list)

    mean_width = np.mean(width_array)
    std_width = np.std(width_array)

    # Smaller value means more uniform
    uniformity_index = std_width / mean_width

    # Use 10th percentile instead of minimum to reduce noise effect
    resolution_metric = np.percentile(width_array, 10)

    #Step 12: Caclulate score
    target_resolution = 40
    # penalties
    penalty_uniformity = 10 * uniformity_index
    penalty_breakpoints = 40 * breakpoints
    penalty_resolution = 0.5 * abs(resolution_metric - target_resolution)

    # total score
    score = 100 - penalty_uniformity - penalty_breakpoints - penalty_resolution

    # keep score in range 0 to 100
    score = max(0, min(100, score))


    #Step 13: Print the results
    output = image.copy()
       # Draw scan lines on the output image
    if scan_direction == "horizontal":
        for y_pos in scan_positions:
            cv2.line(output, (x, y_pos), (x + w - 1, y_pos), (255, 0, 0), 1)

    elif scan_direction == "vertical":
        for x_pos in scan_positions:
            cv2.line(output, (x_pos, y), (x_pos, y + h - 1), (255, 0, 0), 1)

    cv2.drawContours(output, valid_contours, -1, (0, 255, 0), 2)

    #show the results on the image
    cv2.putText(output, f"Mean width: {mean_width:.2f}px", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(output, f"Uniformity: {uniformity_index:.3f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(output, f"Breakpoints: {breakpoints}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(output, f"Resolution metric: {resolution_metric:.2f}px", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(output, f"Score: {score:.2f}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    #Print the result
    print("===== Analysis Results =====")
    print("Average line width =", mean_width)
    print("Width std =", std_width)
    print("Uniformity index =", uniformity_index)
    print("Breakpoints =", breakpoints)
    print("Resolution metric =", resolution_metric)
    
    print("\n===== Score Calculation =====")
    print("Final score =", score)

    #save the images
    cv2.imwrite("01_gray.jpg", gray)
    cv2.imwrite("02_binary.jpg", binary)
    cv2.imwrite("03_mask.jpg", mask)
    cv2.imwrite("04_result.jpg", output)

    print("Images saved:")
    print("01_gray.jpg")
    print("02_binary.jpg")
    print("03_mask.jpg")
    print("04_result.jpg")

    return {"score": score}