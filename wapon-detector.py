import pyautogui
import numpy as np
import easyocr
import keyboard
import cv2  # OpenCV for image processing
import os  # For file operations

def capture_and_ocr():
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()

    # Calculate dimensions for the bottom-right 30% region
    width = int(screen_width * 0.1)
    height = int(screen_height * 0.3)
    left = int(screen_width - width)
    top = int(screen_height - height)

    # Capture the screenshot of the specified region
    screenshot = pyautogui.screenshot(region=(left, top, width, height))

    # Convert to numpy array (BGR format for OpenCV)
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    # Crop 5% from the bottom after capturing the 30% region
    crop_height = int(height * 0.70)  # Keep 95% (remove bottom 5%)
    screenshot_np = screenshot_np[:crop_height, :]  # Crop bottom 5%

    # Define image filename (overwrite the previous one)
    image_path = "captured_image.png"

    # Remove the previous image if it exists
    if os.path.exists(image_path):
        os.remove(image_path)

    # Save the new image
    cv2.imwrite(image_path, screenshot_np)
    print(f"Screenshot saved as {image_path}")

    # Initialize EasyOCR reader with GPU support
    reader = easyocr.Reader(['en'], gpu=True)

    # Perform OCR on the cropped screenshot
    results = reader.readtext(screenshot_np)

    # Print the extracted text
    for bbox, text, prob in results:
        print(f'Text: {text}, Probability: {prob}')

# Bind the F6 key to the capture_and_ocr function
keyboard.add_hotkey('f6', capture_and_ocr)

# Keep the script running to listen for the key press
keyboard.wait('esc')
