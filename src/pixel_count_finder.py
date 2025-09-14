import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2


def active_pixels_count(image):
    lower_black = np.array([0, 0, 0], dtype="uint8")
    upper_black = np.array([0, 0, 0], dtype="uint8")

    # Create a mask for black pixels
    mask = cv2.inRange(image, lower_black, upper_black)

    # Count the number of non-zero pixels in the mask (which are the black pixels)
    black_pixel_count = cv2.countNonZero(mask)

    return (image.shape[0]*image.shape[1]) - black_pixel_count

def detect_red_zones(image):   
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red (both ends of the HSV spectrum)
    lower_red1 = np.array([0, 50, 50])       # Darker red shades
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 50])     # Lighter red shades
    upper_red2 = np.array([180, 255, 255])

    # Combine both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the original image
    red_zone = cv2.bitwise_and(image, image, mask=mask)
    # Save the result as a new image
    output_filename = "data/red_zones_detected.jpg"
    cv2.imwrite(output_filename, red_zone)
    return red_zone,active_pixels_count(red_zone)


def detect_yellow_zones(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the yellow spectrum range (covering light to deep yellow)
    lower_yellow = np.array([20, 50, 50])    # Lighter yellow shades
    upper_yellow = np.array([40, 255, 255])  # Darker yellow shades

    # Create the mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the mask to the original image
    yellow_zone = cv2.bitwise_and(image, image, mask=mask)

    # Save the result as a new image
    output_filename = "data/yellow_zones_detected.jpg"
    cv2.imwrite(output_filename, yellow_zone)
    return yellow_zone,active_pixels_count(yellow_zone)

    
def detect_green_zones(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the green spectrum range
    lower_green = np.array([35, 50, 50])    # Light green shades
    upper_green = np.array([85, 255, 255])  # Dark green shades

    # Create the mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply the mask to the original image
    green_zone = cv2.bitwise_and(image, image, mask=mask)

    # Save the result as a new image
    output_filename = "data/green_zones_detected.jpg"
    cv2.imwrite(output_filename, green_zone)

    print(f"Green zones saved as '{output_filename}'")
    return green_zone,active_pixels_count(green_zone)

def detect_blue_zones(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the blue spectrum range
    lower_blue = np.array([90, 50, 50])    # Light blue shades
    upper_blue = np.array([130, 255, 255])  # Dark blue shades

    # Create the mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the original image
    blue_zone = cv2.bitwise_and(image, image, mask=mask)

    # Save the result as a new image
    output_filename = "data/blue_zones_detected.jpg"
    cv2.imwrite(output_filename, blue_zone)

    print(f"Blue zones saved as '{output_filename}'")
    return blue_zone,active_pixels_count(blue_zone)

