import cv2
import time
import numpy as np
import ctypes
import win32api
import threading
import bettercam
from multiprocessing import Queue, Process
from ctypes import windll
import os
import json
import random
import signal
import sys

# Global shutdown event for graceful termination
shutdown_event = threading.Event()

def cls():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def simulate_shoot(q):
    """
    Process function to simulate a shooting key press.
    Adds randomization to mimic human-like behavior.
    """
    keybd_event = windll.user32.keybd_event
    while not shutdown_event.is_set():
        try:
            signal_value = q.get(timeout=0.1)
        except Exception:
            continue

        if signal_value == "Shoot":
            press_duration = random.uniform(0.05, 0.15)  # Realistic human-like delay
            keybd_event(0x01, 0, 0, 0)  # Press Left Mouse Button
            time.sleep(press_duration)
            keybd_event(0x01, 0, 2, 0)  # Release Left Mouse Button

def detect_color(frame, cmin, cmax):
    """Returns True if the target color is detected in the frame."""
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, cmin, cmax)
        return np.any(mask)
    except cv2.error:
        return False

class Triggerbot:
    def __init__(self, q, keybind, fov, hsv_range, shooting_rate, fps):
        self.queue = q
        self.keybind = keybind
        self.shooting_rate = shooting_rate / 1000.0  # Convert ms to seconds
        self.fps = int(fps)
        self.fov = int(fov)
        self.last_shot_time = 0

        user32 = windll.user32
        self.WIDTH, self.HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2

        self.check_region = (
            max(0, center_x - self.fov),
            max(0, center_y - self.fov),
            min(self.WIDTH, center_x + self.fov),
            min(self.HEIGHT, center_y + self.fov),
        )

        self.camera = bettercam.create(output_idx=0, region=self.check_region)
        self.camera.start(target_fps=self.fps)

        self.cmin = np.array(hsv_range[0], dtype=np.uint8)
        self.cmax = np.array(hsv_range[1], dtype=np.uint8)

    def run(self):
        while not shutdown_event.is_set():
            if win32api.GetAsyncKeyState(self.keybind) < 0:
                frame = self.camera.get_latest_frame()
                if frame is not None and detect_color(frame, self.cmin, self.cmax):
                    current_time = time.time()
                    if current_time - self.last_shot_time >= self.shooting_rate:
                        self.queue.put("Shoot")
                        self.last_shot_time = current_time
                time.sleep(1 / self.fps)
            else:
                time.sleep(0.01)

def save_config(config, filename='config.json'):
    try:
        with open(filename, 'w') as config_file:
            json.dump(config, config_file, indent=4)
    except Exception as e:
        print("Error saving config:", e)

def load_config(filename='config.json'):
    try:
        with open(filename, 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        print("Error loading config:", e)
        return {}

def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    shutdown_event.set()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    shoot_queue = Queue()
    shoot_process = Process(target=simulate_shoot, args=(shoot_queue,))
    shoot_process.start()

    config_path = 'config.json'
    config = load_config(config_path) if os.path.exists(config_path) else {}

    triggerbot = Triggerbot(
        shoot_queue,
        config.get('keybind', 164),
        config.get('fov', 5.0),
        config.get('hsv_range', [[30, 125, 150], [30, 255, 255]]),
        config.get('shooting_rate', 65.0),
        config.get('fps', 200.0)
    )

    triggerbot_thread = threading.Thread(target=triggerbot.run, daemon=True)
    triggerbot_thread.start()

    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        shutdown_event.set()

    shoot_process.terminate()
    shoot_process.join()
