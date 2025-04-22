import os
import time
import threading
import random
import numpy as np
import keyboard
import mss
import cv2
import win32api
import sys
from pynput.mouse import Listener
import torch
import queue
from pathlib import Path
from shared_scanner import SharedScanner, ScanningManager
import ctypes
import pyautogui

print("Initializing Aimlock...")

# Global variables with thread-safe mechanisms
is_active = False
last_active = None
last_target_found = None
last_mode = None
last_device = None
target_found = False
detection_time = 0
average_response_time = 0
should_exit = False

detection_queue = queue.Queue(maxsize=3)

class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()
    def increment(self, delta=1):
        with self._lock:
            self._value += delta
            return self._value
    def get(self):
        with self._lock:
            return self._value
    def set(self, value):
        with self._lock:
            self._value = value
            return self._value

detection_counter = AtomicCounter(0)
frames_processed = AtomicCounter(0)

config = {
    "target_color": "yellow",
    "tolerance": 45,
    "scan_area_x": 16,
    "scan_area_y": 10,
    "aim_lock_strength": 68,
    "toggle_key": "alt",
    "refresh_rate": 240,
    "debug_mode": False,
    "use_gpu": torch.cuda.is_available(),
    "aim_lock_mode": "hold",  # or "toggle"
}

BGR_TARGET = (64, 254, 254)

SCREENSHOT_REGION = None
hook_status_path = Path("aimlock/hook_status.txt")

# --- Logging ---
LOG_LEVELS = {"INFO": 1, "DEBUG": 2, "WARN": 3, "ERROR": 4}
LOG_HISTORY = {"last": None}
def log_event(msg, level="INFO", context=None, dedup_key=None):
    """Log with level, context, and deduplication."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    prefix = f"[{timestamp}] [{level}]"
    if context:
        prefix += f" [{context}]"
    line = f"{prefix} {msg}"
    if dedup_key:
        if LOG_HISTORY.get(dedup_key) == line:
            return
        LOG_HISTORY[dedup_key] = line
    # Only print if called from inside AimLockController (not globally)
    # print(line)

def write_hook_status(active):
    try:
        with open(hook_status_path, "w") as f:
            f.write("active" if active else "inactive")
    except Exception:
        pass

def initialize_screenshot_region():
    global SCREENSHOT_REGION
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screen_width = monitor['width']
        screen_height = monitor['height']
        center_x = screen_width // 2
        center_y = screen_height // 2
        SCREENSHOT_REGION = {
            'left': center_x - config["scan_area_x"] // 2,
            'top': center_y - config["scan_area_y"] // 2,
            'width': config["scan_area_x"],
            'height': config["scan_area_y"]
        }

class AimLockController:
    def __init__(self, config_override=None):
        self.config = config.copy()
        if config_override:
            self.config.update(config_override)
            
        # CRITICAL FIX: If aim_lock_toggle_key is missing or doesn't match aim_lock_keybind, regenerate it
        key_code = self.config.get('aim_lock_keybind', 164)
        key_str = self.config.get('aim_lock_toggle_key', None)
        
        # Map key codes to names
        inv_key_map = {
            164: "alt", 160: "shift", 20: "caps lock", 9: "tab", 
            0x58: "x", 0x43: "c", 0x5A: "z", 0x56: "v",
            0x02: "mouse right", 0x04: "mouse 3", 0x05: "mouse 4", 0x06: "mouse 5"
        }
        
        # If key_code is present, ensure key_str matches
        if key_code in inv_key_map:
            correct_key_str = inv_key_map[key_code].lower()
            # If aim_lock_toggle_key is missing or wrong, fix it
            if not key_str or key_str.lower() != correct_key_str:
                self.config['aim_lock_toggle_key'] = correct_key_str
                key_str = correct_key_str
                
        # Debug print the keys being loaded
        print(f"[DEBUG] AimLock initialized with key: {self.config.get('aim_lock_toggle_key', 'alt')} (code: {key_code})")
        
        self.is_active = False
        self.target_found = False
        self.should_exit = False
        self.status = {
            'active': False,
            'target_found': False,
            'average_response_time': 0,
            'fps': 0
        }
        self.toggle_thread = None
        self.status_thread = None
        self.mouse_listener = None
        self.manager = ScanningManager.get_instance(fps=self.config.get('refresh_rate', 240))
        self.region_name = f"aimlock_{id(self)}"
        self.last_detected = None
        self.last_log_time = 0
        self.color_detected_count = 0
        self.frame_times = []
        self.max_frame_history = 30
        user32 = ctypes.windll.user32
        WIDTH, HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        fov_size = 8
        left = center_x - fov_size
        top = center_y - fov_size
        right = center_x + fov_size
        bottom = center_y + fov_size
        left = max(0, left)
        top = max(0, top)
        right = min(WIDTH, right)
        bottom = min(HEIGHT, bottom)
        global SCREENSHOT_REGION
        SCREENSHOT_REGION = {
            'left': left,
            'top': top,
            'width': right - left,
            'height': bottom - top
        }
        hsv_lower = [30, 125, 150]
        hsv_upper = [30, 255, 255]
        scan_params = {
            'mode': 'gpu' if self.config['use_gpu'] else 'cpu',
            'color_range': (hsv_lower, hsv_upper),
            'tolerance': 45,
            'color_space': 'hsv',
        }
        def detection_callback(detected, frame, region_name):
            if not self.is_active or self.should_exit:
                return
            now = time.perf_counter()
            if detected != self.last_detected:
                write_hook_status(self.is_active and detected)
                log_event(f"Target {'acquired' if detected else 'lost'}.", "INFO", context=f"[AIMLOCK] DEVICE={'GPU' if self.config['use_gpu'] else 'CPU'}", dedup_key="detection")
                self.last_detected = detected
            self.target_found = detected
            self.status['target_found'] = detected
            if detected:
                self.color_detected_count += 1
                self._aim_action()
            frame_time = (now - getattr(self, '_last_frame_time', now)) * 1000
            self._last_frame_time = now
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_history:
                self.frame_times.pop(0)
            self.status['average_response_time'] = int(sum(self.frame_times) / len(self.frame_times))
            self.status['fps'] = int(1000 / self.status['average_response_time']) if self.status['average_response_time'] > 0 else 0
        self._on_detection_result = detection_callback
        self.manager.register_subscriber(
            self.region_name,
            SCREENSHOT_REGION,
            scan_params,
            self._on_detection_result
        )
        self.manager.start()

    def _aim_action(self):
        # No auto-move; sticky aim is handled by mouse listener
        pass

    def _on_mouse_move(self, x, y):
        if self.target_found and self.is_active:
            base_delay = self.config["aim_lock_strength"] / 1000.0
            random_factor = random.uniform(0.9, 1.1)
            micro_delay = base_delay * random_factor
            time.sleep(micro_delay)

    def _toggle_key_loop(self):
        mode = self.config.get("aim_lock_mode", "hold")
        last_key_state = False
        toggled_active = False
        last_key_str = None
        while not self.should_exit:
            try:
                # Always get the latest key from self.config
                key_str = self.config.get("aim_lock_toggle_key", "alt")
                if key_str:
                    # Force lowercase for consistent keyboard detection
                    key_str = key_str.lower()
                
                if key_str != last_key_str:
                    print(f"[DEBUG] AimLock activation key changed to: {key_str}")
                    last_key_str = key_str
                key_state = False
                try:
                    import keyboard
                    key_state = keyboard.is_pressed(key_str)
                except Exception as e:
                    log_event(f"Error in key detection: {e}", "ERROR", context="AIMLOCK")
                if mode == "hold":
                    self.is_active = key_state
                else:  # toggle
                    if key_state and not last_key_state:
                        toggled_active = not toggled_active
                        log_event(f"Toggle mode switched to: {toggled_active}", "INFO", context="AIMLOCK", dedup_key="toggle")
                        time.sleep(0.2)
                    self.is_active = toggled_active
                last_key_state = key_state
                time.sleep(0.01)
            except Exception as e:
                log_event(f"Error in listen_for_key: {e}", "ERROR", context="AIMLOCK")
                time.sleep(0.1)
            
    def _status_loop(self):
        while not self.should_exit:
            status = f"{'ACTIVE' if self.is_active else 'IDLE'} | " \
                     f"Target: {'DETECTED' if self.target_found else 'NOT FOUND'} | " \
                     f"Response: {self.status['average_response_time']}ms | " \
                     f"FPS: {self.status['fps']}"
            time.sleep(0.5)

    def start(self):
        self.should_exit = False
        self.is_active = True
        write_hook_status(False)
        self.toggle_thread = threading.Thread(target=self._toggle_key_loop, name="ToggleThread")
        self.toggle_thread.daemon = True
        self.status_thread = threading.Thread(target=self._status_loop, name="StatusThread")
        self.status_thread.daemon = True
        self.toggle_thread.start()
        self.status_thread.start()
        self.config["aim_lock_enabled"] = True
        # Start mouse listener for sticky aim
        self.mouse_listener = Listener(on_move=self._on_mouse_move)
        self.mouse_listener.start()

    def stop(self):
        self.should_exit = True
        self.is_active = False
        self.config["aim_lock_enabled"] = False
        write_hook_status(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.toggle_thread:
            self.toggle_thread.join(timeout=1)
        if self.status_thread:
            self.status_thread.join(timeout=1)
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        self.manager.unregister_subscriber(self.region_name, self._on_detection_result)

    def update_config(self, new_config):
        self.config.update(new_config)
        global config
        config.update(new_config)
        initialize_screenshot_region()
        # Use HSV detection, matching TriggerBot
        hsv_lower = [30, 125, 150]
        hsv_upper = [30, 255, 255]
        scan_params = {
            'mode': 'gpu' if self.config['use_gpu'] else 'cpu',
            'color_range': (hsv_lower, hsv_upper),
            'tolerance': 45,
            'color_space': 'hsv',
        }
        self.manager.update_region(self.region_name, region_dict=SCREENSHOT_REGION, scan_params=scan_params)

    def get_status(self):
        return self.status.copy()

# Remove global update_status_line, print_static_info, update_status, listen_for_key, track_targets, on_mouse_move
# Only keep AimLockController and its methods
# Remove if __name__ == "__main__" block 