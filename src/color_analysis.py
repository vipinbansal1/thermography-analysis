import cv2
import numpy as np

def get_color_densities(frame):
    """
    Calculates the number of pixels in a frame that fall within the HSV color ranges 
    corresponding to red, yellow, green, and blue.

    Parameters:
        frame (np.ndarray): The input frame in BGR format (OpenCV default).

    Returns:
        Tuple[int, int, int, int]: A tuple containing pixel counts for red, yellow, green, and blue colors.
    """
    # Convert the BGR image to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for red (two ranges due to HSV wrap-around)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    # Define HSV range for yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # Define HSV range for green
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    # Define HSV range for blue
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([130, 255, 255])

    # Generate masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Count the number of non-zero pixels (i.e., pixels within the range)
    red_count = np.sum(red_mask > 0)
    yellow_count = np.sum(yellow_mask > 0)
    green_count = np.sum(green_mask > 0)
    blue_count = np.sum(blue_mask > 0)

    return red_count, yellow_count, green_count, blue_count