import sys, os, time, random, ctypes, json, threading
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2, numpy as np, pyautogui
import win32api, win32con
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QTabWidget, QSplitter, QStatusBar, QToolTip, QStyleFactory, QFileDialog, QInputDialog
import bettercam
from tempfile import gettempdir
from PIL import Image
import traceback
import queue
import easyocr
import mss

# Import the triggerbot module
from triggerbot import (
    TriggerBotThread, 
    simulate_shoot, 
    detect_color, 
    detect_color_pytorch, 
    fast_benchmark, 
    TORCH_AVAILABLE, 
    shutdown_event
)

# Import AimLockController from aimlock.py
from aimlock import AimLockController

# Enable debug mode
DEBUG_MODE = True

def debug_log(message):
    """Log debug messages to console if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}")

# Import PyTorch for better GPU support
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        debug_log(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        debug_log("PyTorch CUDA not available, using CPU only.")
except ImportError:
    TORCH_AVAILABLE = False
    debug_log("PyTorch not installed, using CPU only.")

# Global shutdown event used for both triggerbot and scanning logic
shutdown_event = threading.Event()

# ------------------- Utility Functions -------------------

def save_config(config, filename='config.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print("Error saving config:", e)

def load_config(filename='config.json'):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                # Backward compatibility: map old keys to new ones
                key_map = {
                    "min_shooting_rate": "min_shooting_delay_ms",
                    "max_shooting_rate": "max_shooting_delay_ms",
                    "enable_press_duration": "enable_random_press_duration",
                    "press_duration_min": "press_duration_min_s",
                    "press_duration_max": "press_duration_max_s",
                }
                for old, new in key_map.items():
                    if old in config and new not in config:
                        config[new] = config[old]
                # Remove deprecated keys
                for deprecated in ["shooting_rate", "auto_fallback_to_cpu", "aim_lock_toggle_key"]:
                    if deprecated in config:
                        del config[deprecated]
                # Ensure aim_lock_target_color is always an array
                if "aim_lock_target_color" in config and not isinstance(config["aim_lock_target_color"], list):
                    config["aim_lock_target_color"] = [30, 255, 255]
                # Ensure all new aim lock keys are present
                defaults = {
                    "aim_lock_enabled": False,
                    "aim_lock_target_color": [30, 255, 255],
                    "aim_lock_tolerance": 45,
                    "aim_lock_scan_area_x": 16,
                    "aim_lock_scan_area_y": 10,
                    "aim_lock_strength": 68,
                    "aim_lock_refresh_rate": 240,
                    "aim_lock_debug_mode": False,
                    "aim_lock_adaptive_scan": True,
                    "aim_lock_scan_pattern": "spiral"
                }
                for k, v in defaults.items():
                    if k not in config:
                        config[k] = v
                return config
        except Exception:
            pass
    # Default configuration with new schema
    return {
        "fov": 5.0,
        "keybind": 164,
        "min_shooting_delay_ms": 50.0,  # Minimum shooting delay in ms
        "max_shooting_delay_ms": 80.0,  # Maximum shooting delay in ms
        "fps": 200.0,
        "hsv_range": [[30, 125, 150], [30, 255, 255]],
        "trigger_mode": "hold",  # or "toggle"
        "enable_random_press_duration": True,  # Toggle for using random press duration
        "press_duration_min_s": 0.01,  # Minimum press duration in seconds
        "press_duration_max_s": 0.05,  # Maximum press duration in seconds
        "block_movements": False,   # If True, block (W, A, S, D) while shooting
        "use_gpu": False,  # Use GPU if available
        "smart_acceleration": True,  # Automatically select best acceleration method
        "test_mode": False,  # Test mode for comparing GPU/CPU
        "theme": "custom",  # Default theme set to custom green
        # Aim Lock defaults
        "aim_lock_enabled": False,
        "aim_lock_target_color": [30, 255, 255],
        "aim_lock_tolerance": 45,
        "aim_lock_scan_area_x": 16,
        "aim_lock_scan_area_y": 10,
        "aim_lock_strength": 68,
        "aim_lock_refresh_rate": 240,
        "aim_lock_debug_mode": False,
        "aim_lock_adaptive_scan": True,
        "aim_lock_scan_pattern": "spiral"
    }

def robust_load_config(filename='config.json'):
    key_map = {
        "Alt": 164, "Shift": 160, "Caps Lock": 20, "Tab": 9, "X": 0x58, "C": 0x43, "Z": 0x5A, "V": 0x56,
        "Mouse Right": 0x02, "Mouse 3": 0x04, "Mouse 4": 0x05, "Mouse 5": 0x06
    }
    inv_key_map = {v: k for k, v in key_map.items()}
    config = load_config(filename)
    key_code = config.get('aim_lock_keybind', 164)
    key_str = config.get('aim_lock_toggle_key', None)
    if key_str is None and key_code is not None:
        key_str = inv_key_map.get(key_code, 'Alt')
        config['aim_lock_toggle_key'] = key_str
        save_config(config, filename)
    elif key_code is None and key_str is not None:
        key_code = key_map.get(key_str, 164)
        config['aim_lock_keybind'] = key_code
        save_config(config, filename)
    elif key_str is not None and key_code is not None:
        if key_map.get(key_str, 164) != key_code:
            config['aim_lock_keybind'] = key_map.get(key_str, 164)
            save_config(config, filename)
    return config

# --- PATCH: Alias robust_load_config to load_config before MainWindow ---
robust_load_config = load_config

# ------------------- TriggerBot Logic -------------------

# Triggerbot functions were moved to triggerbot.py

# Add a benchmark function to compare GPU vs CPU performance
def benchmark_gpu_vs_cpu(frame):
    """Run a benchmark to compare CPU vs GPU performance on the same operations"""
    try:
        debug_log("Starting GPU vs CPU benchmark...")
        
        # Create a larger workload for more accurate measurement
        test_iterations = 100  # Increased iterations for more accuracy
        
        # Create a much larger frame to better demonstrate GPU advantage
        large_size = 1920  # Full HD resolution will better show GPU advantage
        if frame is not None:
            # Create a large frame by repeating the input frame
            frame_large = np.tile(frame, (5, 5, 1))
            # Resize to exact dimensions if needed
            h, w = frame_large.shape[:2]
            if h < large_size or w < large_size:
                frame_large = cv2.resize(frame_large, (large_size, large_size))
        else:
            # Create a synthetic test frame with complex patterns for better testing
            frame_large = np.zeros((large_size, large_size, 3), dtype=np.uint8)
            # Add some patterns to make it more challenging
            for i in range(0, large_size, 10):
                frame_large[i:i+5, :, 0] = 255  # Red stripes
            for i in range(0, large_size, 15):
                frame_large[:, i:i+5, 1] = 255  # Green stripes
            # Add some diagonal elements
            for i in range(large_size):
                if i < large_size - 2:
                    frame_large[i, i, 2] = 255
                    frame_large[i+1, i, 2] = 255
                    frame_large[i+2, i, 2] = 255
        
        # Dictionary to store results
        results = {
            "cpu_time": 0.0001,  # Small epsilon to prevent division by zero
            "pytorch_gpu_time": float('inf'),
            "best_gpu_time": float('inf'),
            "ratio": 0,
            "best_method": "CPU"
        }
        
        # PyTorch GPU test
        if TORCH_AVAILABLE and torch.cuda.is_available():
            debug_log("Running PyTorch GPU benchmark...")
            # Force clean CUDA memory
            torch.cuda.empty_cache()
            
            # Extended warmup (crucial for accurate GPU benchmarking)
            for _ in range(10):  # Multiple warmup iterations
                test_frame = torch.from_numpy(frame_large).cuda().float() / 255.0
                test_frame = test_frame.permute(2, 0, 1)
                r, g, b = test_frame[0], test_frame[1], test_frame[2]
                max_val, _ = torch.max(test_frame, dim=0)
                min_val, _ = torch.min(test_frame, dim=0)
                diff = max_val - min_val
                # Force computation
                torch.cuda.synchronize()
                # Clean up
                del test_frame, r, g, b, max_val, min_val, diff
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure GPU is ready
            
            # More intensive benchmark
            pytorch_gpu_start = time.time()
            
            with torch.amp.autocast('cuda', enabled=True):  # Using updated API syntax
                for _ in range(test_iterations):
                    # Convert frame to tensor on GPU
                    frame_tensor = torch.from_numpy(frame_large).cuda().float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                    
                    # Extract RGB channels
                    r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]
                    
                    # Calculate HSV
                    max_val, _ = torch.max(frame_tensor, dim=0)
                    min_val, _ = torch.min(frame_tensor, dim=0)
                    diff = max_val - min_val
                    v = max_val
                    s = torch.where(max_val != 0, diff / max_val, torch.zeros_like(max_val))
                    h = torch.zeros_like(max_val)
                    
                    # Calculate hue
                    r_max_mask = (r == max_val) & (diff != 0)
                    g_max_mask = (g == max_val) & (diff != 0)
                    b_max_mask = (b == max_val) & (diff != 0)
                    
                    h[r_max_mask] = (60 * ((g[r_max_mask] - b[r_max_mask]) / diff[r_max_mask]) + 360) % 360
                    h[g_max_mask] = (60 * ((b[g_max_mask] - r[g_max_mask]) / diff[g_max_mask]) + 120)
                    h[b_max_mask] = (60 * ((r[b_max_mask] - g[b_max_mask]) / diff[b_max_mask]) + 240)
                    
                    # Apply multiple color range checks to truly stress the GPU
                    # This is where the GPU should shine - parallel processing many operations
                    for hue in range(0, 360, 10):  # More color checks (36 instead of 12)
                        h_min, h_max = hue, (hue + 10) % 360
                        mask = (h >= h_min/2) & (h <= h_max/2) & (s >= 50) & (v >= 50)
                        # Apply more operations to the mask
                        area_sum = torch.sum(mask.float())
                        if area_sum > 0:
                            # More operations that benefit from GPU
                            weighted_v = v * mask.float()
                            avg_brightness = torch.sum(weighted_v) / area_sum
                    
                    # Force CUDA to finish computation
                    torch.cuda.synchronize()
            
            pytorch_gpu_time = (time.time() - pytorch_gpu_start) * 1000 / test_iterations
            debug_log(f"PyTorch GPU average time: {pytorch_gpu_time:.2f}ms")
            results["pytorch_gpu_time"] = max(pytorch_gpu_time, 0.0001)  # Prevent zero
            
            # Clean up
            torch.cuda.empty_cache()
        else:
            debug_log("PyTorch GPU benchmark skipped: CUDA not available")
        
        # CPU benchmark (same as the GPU operations but on CPU)
        debug_log("Running CPU benchmark...")
        cpu_start = time.time()
        for _ in range(test_iterations):
            # Convert to HSV on CPU
            hsv = cv2.cvtColor(frame_large, cv2.COLOR_BGR2HSV)
            
            # Loop through hue ranges (same as GPU)
            for h in range(0, 180, 5):
                lower = np.array([h, 50, 50], dtype=np.uint8)
                upper = np.array([h+5, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                count = cv2.countNonZero(mask)
        
        cpu_time = (time.time() - cpu_start) * 1000 / test_iterations
        debug_log(f"CPU average time: {cpu_time:.2f}ms")
        results["cpu_time"] = max(cpu_time, 0.0001)  # Prevent zero
        
        # Determine best GPU time and ratio
        best_gpu_time = min(results["pytorch_gpu_time"], results["cpu_time"])
        results["best_gpu_time"] = best_gpu_time
        results["ratio"] = results["cpu_time"] / best_gpu_time if best_gpu_time < float('inf') else 0
        
        # Determine best method based on ratio
        if results["ratio"] > 1.0:
            # GPU is faster, determine which one
            if results["pytorch_gpu_time"] <= results["cpu_time"]:
                results["best_method"] = "GPU-PyTorch"
            else:
                results["best_method"] = "CPU"
        else:
            results["best_method"] = "CPU"
        
        debug_log(f"Benchmark completed - CPU: {results['cpu_time']:.2f}ms, Best GPU: {best_gpu_time:.2f}ms, Ratio: {results['ratio']:.2f}x, Best method: {results['best_method']}")
        
        return results
        
    except Exception as e:
        debug_log(f"Error during benchmark: {str(e)}")
        traceback.print_exc()
        return {"cpu_time": 0.0001, "pytorch_gpu_time": float('inf'), "best_gpu_time": float('inf'), 
                "ratio": 0, "best_method": "CPU (benchmark failed)"}

# TriggerBot related functions have been moved to triggerbot.py

# ------------------- New Agent Locking (Instalock) Logic -------------------
# UI-DO-NOT-OBFUSCATE-START
def initialize_client(region):
    try:
        from valclient.client import Client
        client = Client(region=region)
        client.activate()
        return region, client
    except Exception as e:
        print(f"Failed to initialize client for region {region}: {e}")
        return region, None

def monitor_region(region, client, agent_uuid, agent_name, seenMatches, seen_lock, done_event, log_func):
    while not done_event.is_set():
        try:
            presence = client.fetch_presence(client.puuid)
            sessionState = presence.get('sessionLoopState')
            if sessionState == "PREGAME":
                match_info = client.pregame_fetch_match()
                match_id = match_info.get('ID')
                with seen_lock:
                    if match_id in seenMatches:
                        continue
                    seenMatches.add(match_id)
                log_func(f"Region '{region}' detected PREGAME with match {match_id}")
                client.pregame_select_character(agent_uuid)
                client.pregame_lock_character(agent_uuid)
                log_func(f"{agent_name.capitalize()} Locked Successfully on region '{region}'")
                done_event.set()  # Signal other threads to stop
                return region
        except Exception as e:
            pass
        time.sleep(1)
    return None

class ScanningWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()
    
    def __init__(self, agent_name, agent_uuid, parent=None):
        super().__init__(parent)
        self.agent_name = agent_name.lower()
        self.agent_uuid = agent_uuid

    def run(self):
        try:
            valid_regions = ['na', 'eu', 'latam', 'br', 'ap', 'kr', 'pbe']
            preinit_clients = {}
            with ThreadPoolExecutor(max_workers=len(valid_regions)) as executor:
                futures = {executor.submit(initialize_client, region): region for region in valid_regions}
                for future in as_completed(futures):
                    region, client = future.result()
                    if client is not None:
                        preinit_clients[region] = client
            if not preinit_clients:
                self.log_signal.emit("Bro Start Valorant First")
                return
            self.log_signal.emit("Starting monitoring in all regions concurrently...")
            seenMatches = set()
            seen_lock = threading.Lock()
            done_event = threading.Event()
            with ThreadPoolExecutor(max_workers=len(preinit_clients)) as executor:
                monitor_futures = {
                    executor.submit(monitor_region, region, client, self.agent_uuid, self.agent_name,
                                      seenMatches, seen_lock, done_event, self.log_signal.emit): region
                    for region, client in preinit_clients.items()
                }
                for future in as_completed(monitor_futures):
                    result = future.result()
                    if result is not None:
                        self.log_signal.emit(f"✅ Successfully locked {self.agent_name.capitalize()} on region: {result}")
                        break
            if not done_event.is_set():
                self.log_signal.emit("❌ Instalock timeout or failed.")
        except Exception as e:
            self.log_signal.emit(f"Error during instalock: {e}")
        self.finished_signal.emit()
# ------------------- Modern UI (Redesigned) -------------------

class MainWindow(QtWidgets.QMainWindow):
    profile_shortcut_signal = QtCore.pyqtSignal(str)
    def __init__(self):
        super().__init__()
        # Apply modern style
        QtWidgets.QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        self.key_map = {
            "Alt": 164,
            "Shift": 160,
            "Caps Lock": 20,
            "Tab": 9,
            "X": 0x58,
            "C": 0x43,
            "Z": 0x5A,
            "V": 0x56,
            "Mouse Right": 0x02,
            "Mouse 3": 0x04,
            "Mouse 4": 0x05,
            "Mouse 5": 0x06
        }
        self.config = robust_load_config()
        
        # Ensure aim_lock_toggle_key exists and is lowercase for keyboard detection
        if 'aim_lock_toggle_key' in self.config:
            self.config['aim_lock_toggle_key'] = self.config['aim_lock_toggle_key'].lower()
        
        # Set window title randomly from the provided names list
        names = [
            "Telegram", "WhatsApp", "Discord", "Skype", "Slack", "Zoom", "Signal", 
            "MicrosoftTeams", "GoogleMeet", "Viber", "FacebookMessenger", "WeChat", 
            "Line", "Kik", "Snapchat", "Instagram", "Twitter", "Facebook", "LinkedIn", 
            "Reddit", "TikTok", "Clubhouse", "Mastodon", "Threads", "BeReal", "Spotify", 
            "AppleMusic", "YouTube", "Netflix", "Hulu", "DisneyPlus", "AmazonPrime", 
            "HBOMax", "Twitch", "SoundCloud", "Deezer", "Pandora", "Tidal", "GoogleDrive",
            "GoogleDocs", "Evernote", "Notion", "Trello", "Asana", "Monday", "ClickUp", 
            "Todoist", "OneNote", "Dropbox", "PayPal", "Venmo", "CashApp", "Zelle", 
            "GooglePay", "ApplePay", "Stripe", "Robinhood", "Revolut", "Wise"
        ]
        
        self.setWindowTitle(random.choice(names))
        self.setMinimumSize(800, 600)
        self.scanning_in_progress = False
        self.last_f5 = 0
        self.trigger_bot_thread = None
        self.worker = None
        
        self.aim_lock_controller = AimLockController(self.config)
        # self.aim_lock_status_timer = QtCore.QTimer(self)
        # self.aim_lock_status_timer.timeout.connect(self.update_aim_lock_status_ui)
        # self.aim_lock_status_timer.start(500)
        
        self.init_ui()
        
        # Status bar with GPU info
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Always show CUDA as available since the user has confirmed CUDA 12.4
        self.statusBar.showMessage("CUDA 12.4 GPU Acceleration Available")
            
        # Set theme based on config (now defaulting to custom green)
        self.apply_theme(self.config.get("theme", "custom"))

        # --- FIX: Ensure AimLockController is started if enabled on load ---
        if self.aim_lock_enabled_checkbox.isChecked():
            self.aim_lock_controller.update_config(self.config)
            self.aim_lock_controller.start()

    def init_ui(self):
        # Main widget and layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create header with program name and version
        header = QtWidgets.QFrame()
        header.setFixedHeight(60)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_label = QtWidgets.QLabel("GamerFun Valo Menu V4")
        header_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_layout.addWidget(header_label)
        
        # Add theme switcher
        theme_layout = QtWidgets.QHBoxLayout()
        theme_label = QtWidgets.QLabel("Theme:")
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Custom Green"])
        # Default to Custom Green (index 2)
        theme_index = {"dark": 0, "light": 1, "custom": 2}.get(self.config.get("theme", "custom"), 2)
        self.theme_combo.setCurrentIndex(theme_index)
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        header_layout.addLayout(theme_layout)
        main_layout.addWidget(header)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(False)
        self.tabs.setStyleSheet("QTabBar::tab { min-width: 120px; min-height: 28px; font-size: 14px; }")
        
        # Create tabs
        self.instalock_tab = QtWidgets.QWidget()
        self.triggerbot_tab = QtWidgets.QWidget()
        self.profiling_tab = QtWidgets.QWidget()
        self.aimlock_tab = QtWidgets.QWidget()
        self.settings_tab = QtWidgets.QWidget()
        self.logs_tab = QtWidgets.QWidget()
        
        # Add tabs to widget in improved order
        self.tabs.addTab(self.instalock_tab, "🕹️ Agent Instalock")
        self.tabs.addTab(self.triggerbot_tab, "🎯 TriggerBot")
        self.tabs.addTab(self.profiling_tab, "🗂️ Profiling")
        self.tabs.addTab(self.aimlock_tab, "🎯 Aim Lock")
        self.tabs.addTab(self.settings_tab, "⚙️ Settings")
        self.tabs.addTab(self.logs_tab, "📜 Logs")
        
        main_layout.addWidget(self.tabs)
        
        # Setup each tab
        self.setup_instalock_tab()
        self.setup_triggerbot_tab()
        self.setup_profiling_tab()
        self.setup_aimlock_tab()
        self.setup_settings_tab()
        self.setup_logs_tab()
        
        # Setup hotkey timer
        self.hotkey_timer = QtCore.QTimer(self)
        self.hotkey_timer.timeout.connect(self.check_hotkey)
        self.hotkey_timer.start(50)

        # Apply styles
        self.apply_theme(self.config.get("theme", "dark"))
        
        # Activate triggerbot on startup if checkbox is checked
        QtCore.QTimer.singleShot(500, self.activate_triggerbot_on_startup)
        
        # Set a reasonable default window size
        self.resize(1100, 800)
        self.setMinimumSize(900, 600)

    def activate_triggerbot_on_startup(self):
        """Activate triggerbot on startup if the checkbox is checked"""
        if self.triggerbot_checkbox.isChecked():
            self.toggle_triggerbot(QtCore.Qt.Checked)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #E0E0E0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                border: 2px solid #32CD32;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #32CD32;
                font-size: 16px;
                font-weight: bold;
            }
            QComboBox, QDoubleSpinBox, QCheckBox, QRadioButton {
                background-color: #1E1E1E;
                border: 1px solid #32CD32;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #32CD32;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                color: #121212;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2aa82a;
            }
            QTextEdit {
                background-color: #1A1A1A;
                border: 2px solid #32CD32;
                border-radius: 5px;
            }
        """)

    def on_press_duration_toggle(self, state):
        # Enable or disable press duration spin boxes based on the toggle
        enabled = (state == QtCore.Qt.Checked)
        self.press_duration_min_spin.setEnabled(enabled)
        self.press_duration_max_spin.setEnabled(enabled)
        self.config["enable_random_press_duration"] = enabled
        save_config(self.config)
        
        # Update active triggerbot if running, so changes take effect immediately
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            self.log(f"Random press duration {'enabled' if enabled else 'disabled'} - applied immediately")

    def update_trigger_mode(self):
        if self.toggle_radio.isChecked():
            self.config["trigger_mode"] = "toggle"
            self.log("Trigger mode set to TOGGLE")
        else:
            self.config["trigger_mode"] = "hold"
            self.log("Trigger mode set to HOLD")
        save_config(self.config)

    def update_keybind(self):
        text = self.trigger_key_combo.currentText().strip()
        code = self.key_map.get(text, 164)
        self.config["keybind"] = code
        save_config(self.config)

    def update_config(self):
        """Update configuration and apply changes to running triggerbot if active"""
        # Get the old FOV value before updating config
        old_fov = self.config.get("fov", 5.0)
        new_fov = self.fov_slider.value()
        
        # Update configuration
        self.config["fov"] = new_fov
        self.config["trigger_mode"] = "hold" if self.hold_radio.isChecked() else "toggle"
        
        # Update with min/max shooting rate
        min_rate = self.min_delay_spin.value()
        max_rate = self.max_delay_spin.value()
        
        # Ensure min is not greater than max
        if min_rate > max_rate:
            max_rate = min_rate
            self.max_delay_spin.setValue(max_rate)
        
        self.config["min_shooting_delay_ms"] = min_rate
        self.config["max_shooting_delay_ms"] = max_rate
        self.config["shooting_rate"] = (min_rate + max_rate) / 2  # Average for backward compatibility
        
        self.config["press_duration_min_s"] = self.press_duration_min_spin.value()
        self.config["press_duration_max_s"] = self.press_duration_max_spin.value()
        self.config["enable_random_press_duration"] = self.press_duration_toggle.isChecked()
        self.config["block_movements"] = self.block_movements_checkbox.isChecked()
        
        # Save the updated configuration
        save_config(self.config)
        
        # Check if FOV changed - this needs special handling
        fov_changed = (old_fov != new_fov)
        if fov_changed:
            debug_log(f"FOV changed from {old_fov} to {new_fov}")
        
        # Apply changes to the running triggerbot immediately
        if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            # First, update the thread's own settings
            self.trigger_bot_thread.fov = int(new_fov)  # Ensure it's an integer
            
            # Update the triggerbot object via its update_config method
            self.trigger_bot_thread.update_config(self.config)
            
            # Force immediate update if FOV changed
            if fov_changed and hasattr(self.trigger_bot_thread, 'triggerbot') and self.trigger_bot_thread.triggerbot:
                triggerbot = self.trigger_bot_thread.triggerbot
                
                # Force FOV update explicitly - ensure it's an integer
                triggerbot.fov = int(new_fov)  # Cast to int
                
                # Update the check region based on new FOV
                try:
                    debug_log("FOV changed - forcing immediate check region update")
                    
                    # Calculate the new region based on new FOV
                    center_x, center_y = triggerbot.WIDTH // 2, triggerbot.HEIGHT // 2
                    fov_size = max(4, int(new_fov))  # Ensure fov_size is an integer
                    
                    # Calculate region with explicit int casting
                    left = int(center_x - fov_size)
                    top = int(center_y - fov_size)
                    right = int(center_x + fov_size)
                    bottom = int(center_y + fov_size)
                    
                    # Clamp to screen bounds
                    left = max(0, left)
                    top = max(0, top)
                    right = min(triggerbot.WIDTH, right)
                    bottom = min(triggerbot.HEIGHT, bottom)
                    
                    # Update the check region - properly use the dictionary format expected by ScanningManager
                    new_region = {
                        'left': int(left),
                        'top': int(top),
                        'width': int(right - left),
                        'height': int(bottom - top)
                    }
                    triggerbot.check_region = new_region
                    debug_log(f"New check region: {triggerbot.check_region}")
                    
                    # Update the region in the scanning manager
                    if hasattr(triggerbot, 'manager') and hasattr(triggerbot, 'region_name'):
                        triggerbot.manager.update_region(triggerbot.region_name, region_dict=new_region)
                        debug_log(f"Region updated in scanning manager for {triggerbot.region_name}")
                    
                    self.log(f"FOV updated from {old_fov} to {new_fov} - check region immediately changed")
                except Exception as e:
                    debug_log(f"Error during forced check region update: {e}")
                    traceback.print_exc()
                    self.log("Error updating check region - try restarting the TriggerBot")
            
            self.log("Settings updated and applied to running TriggerBot")
        else:
            self.log("Settings saved (will apply on next TriggerBot start)")

    def update_gpu_config(self):
        # Update GPU config (now inverted - checkbox ON means CPU mode)
        use_gpu = self.use_gpu_checkbox.isChecked()
        self.config["use_gpu"] = use_gpu
        save_config(self.config)
        
        # Update the triggerbot if running
        if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            
            # Force immediate GPU config update
            if hasattr(self.trigger_bot_thread, 'triggerbot') and self.trigger_bot_thread.triggerbot:
                triggerbot = self.trigger_bot_thread.triggerbot
                
                # Signal that config needs to be reloaded on next processing cycle
                triggerbot.config_updated = True
                triggerbot.use_gpu = use_gpu
                
                self.log(f"GPU acceleration {'enabled' if use_gpu else 'disabled'} - applied immediately")
        else:
            self.log(f"GPU acceleration {'enabled' if use_gpu else 'disabled'} - will apply on next start")
        
        # If enabling GPU, suggest running a benchmark
        if use_gpu:
            self.log("Consider running a benchmark to verify GPU performance")

    def change_theme(self):
        theme_index = self.theme_combo.currentIndex()
        theme = ["dark", "light", "custom"][theme_index]
        self.config["theme"] = theme
        save_config(self.config)
        self.apply_theme(theme)
    
    def apply_theme(self, theme):
        if theme == "dark":
            self.setStyleSheet("""
                QWidget {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }
                QGroupBox {
                    border: 1px solid #3a3a3a;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #e0e0e0;
                }
                QPushButton {
                    background-color: #0078d7;
                    border: none;
                    border-radius: 3px;
                    padding: 6px 12px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #0086f0;
                }
                QPushButton:pressed {
                    background-color: #00569c;
                }
                QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #2d2d2d;
                    border: 1px solid #3a3a3a;
                    border-radius: 3px;
                    padding: 4px;
                }
                QTabWidget::pane {
                    border: 1px solid #3a3a3a;
                    border-radius: 3px;
                }
                QTabBar::tab {
                    background-color: #2d2d2d;
                    color: #e0e0e0;
                    border: 1px solid #3a3a3a;
                    border-bottom-color: #3a3a3a;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                    padding: 8px 12px;
                }
                QTabBar::tab:selected {
                    background-color: #0078d7;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #373737;
                }
                QTextEdit {
                    background-color: #2d2d2d;
                    border: 1px solid #3a3a3a;
                    border-radius: 3px;
                }
                QCheckBox::indicator, QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    background-color: #1a1a1a;
                    border: 2px solid #32CD32;
                    border-radius: 3px;
                }
                QRadioButton::indicator {
                    border-radius: 8px;
                }
                QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                    background-color: #32CD32;
                    border: 2px solid white;
                }
                QCheckBox::indicator:unchecked:hover, QRadioButton::indicator:unchecked:hover {
                    border: 2px solid white;
                    background-color: #2a2a2a;
                }
                QStatusBar {
                    background-color: #0078d7;
                    color: white;
                }
            """)
        elif theme == "light":
            self.setStyleSheet("""
                QWidget {
                    background-color: #f0f0f0;
                    color: #202020;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }
                QGroupBox {
                    border: 1px solid #c0c0c0;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #202020;
                }
                QPushButton {
                    background-color: #0078d7;
                    border: none;
                    border-radius: 3px;
                    padding: 6px 12px;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #0086f0;
                }
                QPushButton:pressed {
                    background-color: #00569c;
                }
                QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: white;
                    border: 1px solid #c0c0c0;
                    border-radius: 3px;
                    padding: 4px;
                }
                QTabWidget::pane {
                    border: 1px solid #c0c0c0;
                    border-radius: 3px;
                }
                QTabBar::tab {
                    background-color: #e0e0e0;
                    color: #202020;
                    border: 1px solid #c0c0c0;
                    border-bottom-color: #c0c0c0;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                    padding: 8px 12px;
                }
                QTabBar::tab:selected {
                    background-color: #0078d7;
                    color: white;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #f0f0f0;
                }
                QTextEdit {
                    background-color: white;
                    border: 1px solid #c0c0c0;
                    border-radius: 3px;
                }
                QCheckBox::indicator, QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    background-color: #1a1a1a;
                    border: 2px solid #32CD32;
                    border-radius: 3px;
                }
                QRadioButton::indicator {
                    border-radius: 8px;
                }
                QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                    background-color: #32CD32;
                    border: 2px solid white;
                }
                QCheckBox::indicator:unchecked:hover, QRadioButton::indicator:unchecked:hover {
                    border: 2px solid white;
                    background-color: #2a2a2a;
                }
                QStatusBar {
                    background-color: #0078d7;
                    color: white;
                }
            """)
        else:  # custom (green)
            self.setStyleSheet("""
                QWidget {
                    background-color: #121212;
                    color: #E0E0E0;
                    font-family: 'Segoe UI', sans-serif;
                }
                QGroupBox {
                    border: 2px solid #32CD32;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 15px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #32CD32;
                }
                QPushButton {
                    background-color: #32CD32;
                    border: none;
                    border-radius: 3px;
                    padding: 6px 12px;
                    color: black;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2aa82a;
                }
                QPushButton:pressed {
                    background-color: #228b22;
                }
                QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #1a1a1a;
                    border: 1px solid #32CD32;
                    border-radius: 3px;
                    padding: 4px;
                    color: #FFFFFF;
                }
                QTabWidget::pane {
                    border: 1px solid #32CD32;
                    border-radius: 3px;
                }
                QTabBar::tab {
                    background-color: #1a1a1a;
                    color: #e0e0e0;
                    border: 1px solid #32CD32;
                    border-top-left-radius: 3px;
                    border-top-right-radius: 3px;
                    padding: 8px 12px;
                }
                QTabBar::tab:selected {
                    background-color: #32CD32;
                    color: black;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #222222;
                }
                QTextEdit {
                    background-color: #1a1a1a;
                    border: 1px solid #32CD32;
                    border-radius: 3px;
                }
                QCheckBox, QRadioButton {
                    spacing: 8px;
                    padding: 4px;
                }
                QCheckBox::indicator, QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                    background-color: #1a1a1a;
                    border: 2px solid #32CD32;
                    border-radius: 3px;
                }
                QRadioButton::indicator {
                    border-radius: 9px;
                }
                QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                    background-color: #32CD32;
                    border: 2px solid #FFFFFF;
                }
                QCheckBox::indicator:unchecked:hover, QRadioButton::indicator:unchecked:hover {
                    border: 2px solid #FFFFFF;
                    background-color: #2a2a2a;
                }
                QSpinBox::up-button, QDoubleSpinBox::up-button {
                    background-color: #32CD32;
                    border-top-right-radius: 3px;
                    subcontrol-origin: border;
                    subcontrol-position: top right;
                    width: 18px;
                    height: 12px;
                }
                QSpinBox::down-button, QDoubleSpinBox::down-button {
                    background-color: #32CD32;
                    border-bottom-right-radius: 3px;
                    subcontrol-origin: border;
                    subcontrol-position: bottom right;
                    width: 18px;
                    height: 12px;
                }
                QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                    width: 6px;
                    height: 6px;
                    background: black;
                    border: 1px solid black;
                }
                QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                    width: 6px;
                    height: 6px;
                    background: black;
                    border: 1px solid black;
                }
                QStatusBar {
                    background-color: #32CD32;
                    color: black;
                }
            """)

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.append(f"<span style='color:#32CD32'>{timestamp}</span>{message}")
        # Update status label if this is an instalock message
        if "lock" in message.lower():
            self.status_label.setText(message)

    def clear_logs(self):
        self.log_text.clear()

    def check_hotkey(self):
        if win32api.GetAsyncKeyState(win32con.VK_F5) < 0:
            current_time = time.time()
            if (current_time - self.last_f5) > 1.0:
                self.last_f5 = current_time
                if not self.scanning_in_progress:
                    self.start_scanning_session()
                else:
                    self.log("Scan already in progress")

    def start_scanning_session(self):
        agent_name = self.agent_combo.currentText()
        agent_uuid = self.agent_combo.currentData()
        self.log(f"Starting instalock for: {agent_name}")
        self.status_label.setText(f"Scanning for {agent_name}...")
        self.scanning_in_progress = True
        self.worker = ScanningWorker(agent_name, agent_uuid)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.session_finished)
        self.worker.start()

    def session_finished(self):
        self.scanning_in_progress = False
        self.log("Instalock process completed")
        self.status_label.setText("Ready to instalock")

    def toggle_triggerbot(self, state):
        if state == QtCore.Qt.Checked:
            # Get current configuration
            self.update_keybind()
            self.update_config()
            
            # Check if GPU acceleration was forced to off due to missing CUDA support
            if not TORCH_AVAILABLE and self.config.get("use_gpu", False):
                self.config["use_gpu"] = False
                save_config(self.config)
                self.log("GPU acceleration disabled - PyTorch not installed")
            
            # Use the average of min and max for backward compatibility
            avg_shooting_rate = (self.min_delay_spin.value() + self.max_delay_spin.value()) / 2
            
            # Stop any existing thread first to ensure clean restart
            if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread:
                try:
                    self.trigger_bot_thread.stop()
                    self.trigger_bot_thread.wait()
                    self.trigger_bot_thread = None
                    time.sleep(0.5)  # Small delay to ensure resources are released
                except Exception as e:
                    self.log(f"Error stopping previous thread: {str(e)}")
            
            # Create and start trigger bot thread
            self.trigger_bot_thread = TriggerBotThread(
                settings=self.config,
                fov=self.fov_slider.value(),
                min_shooting_delay_ms=self.config.get("min_shooting_delay_ms", 50.0),
                max_shooting_delay_ms=self.config.get("max_shooting_delay_ms", 80.0),
                fps=self.config.get("fps", 200.0),
                hsv_range=self.config.get("hsv_range", [[30,125,150],[30,255,255]])
            )
            self.trigger_bot_thread.log_signal.connect(self.log)
            self.trigger_bot_thread.start()
            self.log("TriggerBot ACTIVATED")
        else:
            # Stop the trigger bot thread
            if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread:
                try:
                    self.trigger_bot_thread.stop()
                    if hasattr(self.trigger_bot_thread, 'wait'):
                        self.trigger_bot_thread.wait(1000)
                    print('[DEBUG] TriggerBot fully stopped and object deleted')
                    self.trigger_bot_thread = None
                except Exception as e:
                    self.log(f"Error stopping triggerbot: {str(e)}")
            # Check if both features are off
            if (not self.trigger_bot_thread) and (not getattr(self, 'aim_lock_controller', None) or not self.aim_lock_enabled_checkbox.isChecked()):
                print('[DEBUG] All scanning and status updates stopped (TriggerBot)')

    def closeEvent(self, event):
        # Stop TriggerBot thread/process
        if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread:
            try:
                self.trigger_bot_thread.stop()
                if hasattr(self.trigger_bot_thread, 'wait'):
                    self.trigger_bot_thread.wait(1000)
                self.trigger_bot_thread = None
            except Exception as e:
                print(f"Error stopping TriggerBot: {e}")
        # Stop AimLockController if running
        if hasattr(self, 'aim_lock_controller') and self.aim_lock_controller:
            try:
                self.aim_lock_controller.stop()
                self.aim_lock_controller = None
            except Exception as e:
                print(f"Error stopping AimLockController: {e}")
        # Stop any worker threads
        if hasattr(self, 'worker') and self.worker:
            try:
                shutdown_event.set()
                self.worker.quit()
                self.worker.wait(1000)
                self.worker = None
            except Exception as e:
                print(f"Error stopping worker: {e}")
        # Stop any custom threads (e.g., shortcut listener)
        if hasattr(self, 'shortcut_thread') and self.shortcut_thread:
            try:
                self.shortcut_listener_active = False
                if self.shortcut_thread.is_alive():
                    self.shortcut_thread.join(timeout=1.0)
                self.shortcut_thread = None
            except Exception as e:
                print(f"Error stopping shortcut thread: {e}")
        print('[DEBUG] MainWindow closed, all scanning and status updates should be stopped')
        event.accept()

    def setup_instalock_tab(self):
        layout = QtWidgets.QVBoxLayout(self.instalock_tab)
        
        # Warning about ban risk
        warning_group = QtWidgets.QGroupBox("⚠️ WARNING")
        warning_layout = QtWidgets.QVBoxLayout(warning_group)
        
        warning_label = QtWidgets.QLabel("Using the Insta Agent Lock feature carries a HIGH RISK of account suspension or permanent ban. This feature directly interacts with game client APIs in a way that violates Riot's Terms of Service. Use at your own risk.")
        warning_label.setStyleSheet("color: #FF0000; font-weight: bold;")
        warning_label.setWordWrap(True)
        warning_layout.addWidget(warning_label)
        
        layout.addWidget(warning_group)
        
        # Agent selection group
        agent_group = QtWidgets.QGroupBox("Agent Selection")
        agent_layout = QtWidgets.QVBoxLayout(agent_group)
        
        # Agent selection row
        agent_select_layout = QtWidgets.QHBoxLayout()
        agent_label = QtWidgets.QLabel("Select Agent:")
        self.agent_combo = QtWidgets.QComboBox()
        
        static_agents = {
            "gekko": "e370fa57-4757-3604-3648-499e1f642d3f",
            "fade": "dade69b4-4f5a-8528-247b-219e5a1facd6",
            "breach": "5f8d3a7f-467b-97f3-062c-13acf203c006",
            "deadlock": "cc8b64c8-4b25-4ff9-6e7f-37b4da43d235",
            "tejo": "b444168c-4e35-8076-db47-ef9bf368f384",
            "raze": "f94c3b30-42be-e959-889c-5aa313dba261",
            "chamber": "22697a3d-45bf-8dd7-4fec-84a9e28c69d7",
            "kay/o": "601dbbe7-43ce-be57-2a40-4abd24953621",
            "skye": "6f2a04ca-43e0-be17-7f36-b3908627744d",
            "cypher": "117ed9e3-49f3-6512-3ccf-0cada7e3823b",
            "sova": "320b2a48-4d9b-a075-30f1-1f93a9b638fa",
            "killjoy": "1e58de9c-4950-5125-93e9-a0aee9f98746",
            "harbor": "95b78ed7-4637-86d9-7e41-71ba8c293152",
            "vyse": "efba5359-4016-a1e5-7626-b1ae76895940",
            "viper": "707eab51-4836-f488-046a-cda6bf494859",
            "phoenix": "eb93336a-449b-9c1b-0a54-a891f7921d69",
            "astra": "41fb69c1-4189-7b37-f117-bcaf1e96f1bf",
            "brimstone": "9f0d8ba9-4140-b941-57d3-a7ad57c6b417",
            "iso": "0e38b510-41a8-5780-5e8f-568b2a4f2d6c",
            "clove": "1dbf2edd-4729-0984-3115-daa5eed44993",
            "neon": "bb2a4828-46eb-8cd1-e765-15848195d751",
            "yoru": "7f94d92c-4234-0a36-9646-3a87eb8b5c89",
            "waylay": "df1cb487-4902-002e-5c17-d28e83e78588",
            "sage": "569fdd95-4d10-43ab-ca70-79becc718b46",
            "reyna": "a3bfb853-43b2-7238-a4f1-ad90e9e46bcc",
            "omen": "8e253930-4c05-31dd-1b6c-968525494517",
            "jett": "add6443a-41bd-e414-f6ad-e58d267f4e95"
        }
        
        self.agent_combo.clear()
        for agent_name in sorted(static_agents.keys()):
            self.agent_combo.addItem(agent_name.capitalize(), static_agents[agent_name])
        
        agent_select_layout.addWidget(agent_label)
        agent_select_layout.addWidget(self.agent_combo)
        agent_layout.addLayout(agent_select_layout)
        
        # Info label
        info_label = QtWidgets.QLabel("Press F5 before you enter the agent picking phase")
        info_label.setStyleSheet("font-style: italic;")
        info_label.setAlignment(QtCore.Qt.AlignCenter)
        agent_layout.addWidget(info_label)
        
        # Manual trigger button
        trigger_button = QtWidgets.QPushButton("Start Instalock Manually")
        trigger_button.clicked.connect(self.start_scanning_session)
        agent_layout.addWidget(trigger_button)
        
        layout.addWidget(agent_group)
        
        # Current status
        status_group = QtWidgets.QGroupBox("Instalock Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        
        self.status_label = QtWidgets.QLabel("Ready to instalock")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
        
        # Add spacer to push content to top
        layout.addStretch()

    def setup_triggerbot_tab(self):
        layout = QtWidgets.QVBoxLayout(self.triggerbot_tab)
        # TriggerBot control group
        control_group = QtWidgets.QGroupBox("TriggerBot Control")
        control_layout = QtWidgets.QVBoxLayout(control_group)
        # Enable checkbox and key binding
        top_row = QtWidgets.QHBoxLayout()
        self.triggerbot_checkbox = QtWidgets.QCheckBox("Enable TriggerBot")
        self.triggerbot_checkbox.setChecked(True)
        self.triggerbot_checkbox.stateChanged.connect(self.toggle_triggerbot)
        key_layout = QtWidgets.QHBoxLayout()
        key_label = QtWidgets.QLabel("Activation Key:")
        self.trigger_key_combo = QtWidgets.QComboBox()
        self.trigger_key_combo.addItems(list(self.key_map.keys()))
        inv_key_map = {v: k for k, v in self.key_map.items()}
        initial_key = inv_key_map.get(self.config.get("keybind", 164), "Alt")
        self.trigger_key_combo.setCurrentText(initial_key)
        self.trigger_key_combo.currentIndexChanged.connect(self.update_keybind)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.trigger_key_combo)
        top_row.addWidget(self.triggerbot_checkbox)
        top_row.addLayout(key_layout)
        control_layout.addLayout(top_row)
        # Mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Trigger Mode:")
        self.hold_radio = QtWidgets.QRadioButton("Hold")
        self.toggle_radio = QtWidgets.QRadioButton("Toggle")
        if self.config.get("trigger_mode", "hold") == "toggle":
            self.toggle_radio.setChecked(True)
        else:
            self.hold_radio.setChecked(True)
        self.hold_radio.toggled.connect(self.update_trigger_mode)
        self.toggle_radio.toggled.connect(self.update_trigger_mode)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.hold_radio)
        mode_layout.addWidget(self.toggle_radio)
        mode_layout.addStretch()
        control_layout.addLayout(mode_layout)
        layout.addWidget(control_group)
        # --- TriggerBot Settings Group ---
        triggerbot_settings_group = QtWidgets.QGroupBox("TriggerBot Settings")
        triggerbot_settings_layout = QtWidgets.QFormLayout(triggerbot_settings_group)
        # FOV as slider with label, only update config on sliderReleased
        self.fov_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fov_slider.setRange(1, 50)
        self.fov_slider.setValue(int(self.config.get("fov", 5.0)))
        self.fov_slider.setTickInterval(1)
        self.fov_slider.setSingleStep(1)
        self.fov_slider.setToolTip("Field of View: Controls the detection area size.")
        self.fov_slider.sliderReleased.connect(self.update_fov_from_slider)
        self.fov_slider.valueChanged.connect(self.update_fov_label)
        self.fov_value_label = QtWidgets.QLabel(str(self.fov_slider.value()))
        fov_layout = QtWidgets.QHBoxLayout()
        fov_layout.addWidget(self.fov_slider)
        fov_layout.addWidget(self.fov_value_label)
        fov_widget = QtWidgets.QWidget()
        fov_widget.setLayout(fov_layout)
        triggerbot_settings_layout.addRow("Detection FOV:", fov_widget)
        # Shooting delay (min/max)
        self.min_delay_spin = QtWidgets.QDoubleSpinBox()
        self.min_delay_spin.setRange(10.0, 500.0)
        self.min_delay_spin.setValue(self.config.get("min_shooting_delay_ms", 50.0))
        self.min_delay_spin.setSingleStep(5.0)
        self.min_delay_spin.setDecimals(0)
        self.min_delay_spin.valueChanged.connect(self.update_config)
        self.max_delay_spin = QtWidgets.QDoubleSpinBox()
        self.max_delay_spin.setRange(10.0, 500.0)
        self.max_delay_spin.setValue(self.config.get("max_shooting_delay_ms", 80.0))
        self.max_delay_spin.setSingleStep(5.0)
        self.max_delay_spin.setDecimals(0)
        self.max_delay_spin.valueChanged.connect(self.update_config)
        delay_layout = QtWidgets.QHBoxLayout()
        delay_layout.addWidget(QtWidgets.QLabel("Min:"))
        delay_layout.addWidget(self.min_delay_spin)
        delay_layout.addWidget(QtWidgets.QLabel("Max:"))
        delay_layout.addWidget(self.max_delay_spin)
        delay_container = QtWidgets.QWidget()
        delay_container.setLayout(delay_layout)
        triggerbot_settings_layout.addRow("Shooting Delay (ms):", delay_container)
        # Press duration
        self.press_duration_toggle = QtWidgets.QCheckBox("Enable Random Press Duration")
        self.press_duration_toggle.setChecked(self.config.get("enable_random_press_duration", True))
        self.press_duration_toggle.stateChanged.connect(self.on_press_duration_toggle)
        triggerbot_settings_layout.addRow(self.press_duration_toggle)
        self.press_duration_min_spin = QtWidgets.QDoubleSpinBox()
        self.press_duration_min_spin.setRange(0.01, 1.00)
        self.press_duration_min_spin.setValue(self.config.get("press_duration_min_s", 0.01))
        self.press_duration_min_spin.setSingleStep(0.01)
        self.press_duration_min_spin.setDecimals(2)
        self.press_duration_min_spin.valueChanged.connect(self.update_press_duration_min)
        self.press_duration_max_spin = QtWidgets.QDoubleSpinBox()
        self.press_duration_max_spin.setRange(0.01, 1.00)
        self.press_duration_max_spin.setValue(self.config.get("press_duration_max_s", 0.05))
        self.press_duration_max_spin.setSingleStep(0.01)
        self.press_duration_max_spin.setDecimals(2)
        self.press_duration_max_spin.valueChanged.connect(self.update_press_duration_max)
        press_duration_layout = QtWidgets.QFormLayout()
        press_duration_layout.addRow("Min Duration (s):", self.press_duration_min_spin)
        press_duration_layout.addRow("Max Duration (s):", self.press_duration_max_spin)
        triggerbot_settings_layout.addRow(press_duration_layout)
        # Click behavior
        self.block_movements_checkbox = QtWidgets.QCheckBox("Block Movement Keys While Shooting")
        self.block_movements_checkbox.setChecked(self.config.get("block_movements", False))
        self.block_movements_checkbox.stateChanged.connect(self.update_block_movements)
        self.block_movements_checkbox.setToolTip("When enabled, W/A/S/D keys will be temporarily released while shooting")
        triggerbot_settings_layout.addRow(self.block_movements_checkbox)
        layout.addWidget(triggerbot_settings_group)
        layout.addStretch()

    def update_fov_label(self, value):
        self.fov_value_label.setText(str(value))

    def update_fov_from_slider(self):
        self.update_config()

    def setup_settings_tab(self):
        layout = QtWidgets.QVBoxLayout(self.settings_tab)
        # --- Only keep FPS and Debug Mode ---
        perf_group = QtWidgets.QGroupBox("Performance Settings")
        perf_layout = QtWidgets.QFormLayout(perf_group)
        # FPS setting (shared)
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(30.0, 1000.0)
        self.fps_spin.setValue(self.config.get("fps", 200.0))
        self.fps_spin.setSingleStep(10.0)
        self.fps_spin.setDecimals(0)
        self.fps_spin.valueChanged.connect(self.update_config)
        fps_label = QtWidgets.QLabel("Target FPS:")
        fps_note = QtWidgets.QLabel("This controls how often the TriggerBot and Aim Lock scan for targets. Higher FPS = faster response, but more CPU/GPU usage.")
        fps_note.setStyleSheet("font-size: 11px; color: #888;")
        perf_layout.addRow(fps_label, self.fps_spin)
        perf_layout.addRow(fps_note)
        # --- CPU/GPU radio toggle ---
        self.cpu_radio = QtWidgets.QRadioButton("CPU")
        self.gpu_radio = QtWidgets.QRadioButton("GPU")
        use_gpu = self.config.get("use_gpu", False)
        if use_gpu:
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
        self.cpu_radio.toggled.connect(self.update_performance_mode)
        self.gpu_radio.toggled.connect(self.update_performance_mode)
        radio_layout = QtWidgets.QHBoxLayout()
        radio_layout.addWidget(QtWidgets.QLabel("Processing Mode:"))
        radio_layout.addWidget(self.cpu_radio)
        radio_layout.addWidget(self.gpu_radio)
        radio_layout.addStretch()
        perf_layout.addRow(radio_layout)
        layout.addWidget(perf_group)
        # Debug Mode
        debug_group = QtWidgets.QGroupBox("Debug Settings")
        debug_layout = QtWidgets.QVBoxLayout(debug_group)
        self.aim_lock_debug_mode_checkbox = QtWidgets.QCheckBox("Enable Debug Mode (Aim Lock)")
        self.aim_lock_debug_mode_checkbox.setChecked(self.config.get("aim_lock_debug_mode", False))
        self.aim_lock_debug_mode_checkbox.stateChanged.connect(self.update_aim_lock_config)
        debug_layout.addWidget(self.aim_lock_debug_mode_checkbox)
        layout.addWidget(debug_group)
        layout.addStretch()

    def setup_logs_tab(self):
        layout = QtWidgets.QVBoxLayout(self.logs_tab)
        
        # Log viewer
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        clear_button = QtWidgets.QPushButton("Clear Logs")
        clear_button.clicked.connect(self.clear_logs)
        
        controls_layout.addWidget(clear_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.log_text)

    def setup_aimlock_tab(self):
        layout = QtWidgets.QVBoxLayout(self.aimlock_tab)
        aimlock_group = QtWidgets.QGroupBox("Aim Lock Settings")
        aimlock_layout = QtWidgets.QFormLayout(aimlock_group)
        # Master toggle
        self.aim_lock_enabled_checkbox = QtWidgets.QCheckBox("Enable Aim Lock")
        self.aim_lock_enabled_checkbox.setChecked(self.config.get("aim_lock_enabled", False))
        self.aim_lock_enabled_checkbox.stateChanged.connect(self.toggle_aim_lock)
        aimlock_layout.addRow(self.aim_lock_enabled_checkbox)
        # Config controls
        self.aim_lock_radius_x_spin = QtWidgets.QSpinBox()
        self.aim_lock_radius_x_spin.setRange(1, 100)
        self.aim_lock_radius_x_spin.setValue(self.config.get("aim_lock_scan_area_x", 16))
        self.aim_lock_radius_x_spin.valueChanged.connect(self.update_aim_lock_config)
        aimlock_layout.addRow("Scan Area X (px):", self.aim_lock_radius_x_spin)
        self.aim_lock_radius_y_spin = QtWidgets.QSpinBox()
        self.aim_lock_radius_y_spin.setRange(1, 100)
        self.aim_lock_radius_y_spin.setValue(self.config.get("aim_lock_scan_area_y", 10))
        self.aim_lock_radius_y_spin.valueChanged.connect(self.update_aim_lock_config)
        aimlock_layout.addRow("Scan Area Y (px):", self.aim_lock_radius_y_spin)
        # Aim Lock Strength as slider
        self.aim_lock_strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aim_lock_strength_slider.setRange(1, 500)
        self.aim_lock_strength_slider.setValue(self.config.get("aim_lock_strength", 68))
        self.aim_lock_strength_slider.setTickInterval(5)
        self.aim_lock_strength_slider.setSingleStep(1)
        self.aim_lock_strength_slider.setToolTip("Controls how much the aim slows down when a target is detected. Higher = stronger lock.")
        self.aim_lock_strength_slider.sliderReleased.connect(self.update_aim_lock_config)
        self.aim_lock_strength_slider.valueChanged.connect(self.update_aim_lock_strength_label)
        self.aim_lock_strength_value_label = QtWidgets.QLabel(str(self.aim_lock_strength_slider.value()))
        strength_layout = QtWidgets.QHBoxLayout()
        strength_layout.addWidget(self.aim_lock_strength_slider)
        strength_layout.addWidget(self.aim_lock_strength_value_label)
        strength_widget = QtWidgets.QWidget()
        strength_widget.setLayout(strength_layout)
        aimlock_layout.addRow("Aim Lock Strength (ms):", strength_widget)
        # --- Customizable toggle key and mode ---
        # Key selection
        key_layout = QtWidgets.QHBoxLayout()
        key_label = QtWidgets.QLabel("Activation Key:")
        self.aim_lock_key_combo = QtWidgets.QComboBox()
        self.aim_lock_key_combo.addItems(list(self.key_map.keys()))
        # Use the string key if present, else fallback to keybind
        key_str = self.config.get('aim_lock_toggle_key', None)
        if key_str is None:
            inv_key_map = {v: k for k, v in self.key_map.items()}
            key_str = inv_key_map.get(self.config.get('aim_lock_keybind', 164), "Alt")
        self.aim_lock_key_combo.setCurrentText(key_str)
        self.aim_lock_key_combo.currentIndexChanged.connect(self.update_aim_lock_config)
        key_layout.addWidget(key_label)
        key_layout.addWidget(self.aim_lock_key_combo)
        # Mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Activation Mode:")
        self.aim_lock_hold_radio = QtWidgets.QRadioButton("Hold")
        self.aim_lock_toggle_radio = QtWidgets.QRadioButton("Toggle")
        if self.config.get("aim_lock_mode", "hold") == "toggle":
            self.aim_lock_toggle_radio.setChecked(True)
        else:
            self.aim_lock_hold_radio.setChecked(True)
        self.aim_lock_hold_radio.toggled.connect(self.update_aim_lock_config)
        self.aim_lock_toggle_radio.toggled.connect(self.update_aim_lock_config)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.aim_lock_hold_radio)
        mode_layout.addWidget(self.aim_lock_toggle_radio)
        mode_layout.addStretch()
        # Add to form
        aimlock_layout.addRow(key_layout)
        aimlock_layout.addRow(mode_layout)
        layout.addWidget(aimlock_group)
        layout.addStretch()

    def update_aim_lock_strength_label(self, value):
        self.aim_lock_strength_value_label.setText(str(value))

    def update_aim_lock_config(self):
        self.config["aim_lock_scan_area_x"] = self.aim_lock_radius_x_spin.value()
        self.config["aim_lock_scan_area_y"] = self.aim_lock_radius_y_spin.value()
        self.config["aim_lock_strength"] = self.aim_lock_strength_slider.value()
        # Keybind
        key_text = self.aim_lock_key_combo.currentText()
        # Save both the original key name and lowercase version
        self.config["aim_lock_toggle_key"] = key_text.lower() # Always lowercase for keyboard detection
        self.config["aim_lock_keybind"] = self.key_map.get(key_text, 164)
        # Mode
        self.config["aim_lock_mode"] = "toggle" if self.aim_lock_toggle_radio.isChecked() else "hold"
        save_config(self.config)
        print('[DEBUG] Saved aim_lock_toggle_key:', self.config.get('aim_lock_toggle_key'))
        print('[DEBUG] Saved aim_lock_keybind:', self.config.get('aim_lock_keybind'))
        print('[DEBUG] UI combo currentText:', key_text)
        # --- FIX: Do NOT reload config from disk here ---
        # Restart AimLockController with a new instance if enabled
        if self.aim_lock_enabled_checkbox.isChecked():
            if self.aim_lock_controller:
                self.aim_lock_controller.stop()
                del self.aim_lock_controller
            self.aim_lock_controller = AimLockController(self.config)
            self.aim_lock_controller.start()
        else:
            if self.aim_lock_controller:
                self.aim_lock_controller.update_config(self.config)

    def update_aim_lock_status_ui(self):
        status = self.aim_lock_controller.get_status()
        text = f"Status: {'Active' if status['active'] else 'Idle'} | Target: {'Detected' if status['target_found'] else 'Not Found'} | Response: {status['average_response_time']}ms | FPS: {status['fps']}"
        self.aim_lock_status_label.setText(text)

    def update_test_mode(self):
        """Update test mode setting for alternating between CPU and GPU"""
        self.config["test_mode"] = self.test_mode_checkbox.isChecked()
        save_config(self.config)
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            test_mode_str = "ENABLED" if self.test_mode_checkbox.isChecked() else "DISABLED"
            self.log(f"Test mode {test_mode_str} - will alternate between GPU/CPU processing for performance comparison")
            
    def update_fallback_config(self):
        """Update the auto fallback setting in the config"""
        auto_fallback = self.auto_fallback_checkbox.isChecked()
        self.config["auto_fallback_to_cpu"] = auto_fallback
        save_config(self.config)
        
        self.log(f"Auto fallback to CPU {'enabled' if auto_fallback else 'disabled'}")
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
    
    def generate_debug_report(self):
        """Generate and display a debug report for troubleshooting"""
        try:
            self.log("Generating debug report...")
            
            # Create report
            report = []
            report.append("=== GamerFun Valo Menu Debug Report ===")
            report.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"OS: {sys.platform}")
            
            # Python info
            report.append(f"Python: {sys.version}")
            
            # PyTorch info
            report.append("\n=== PyTorch CUDA Support ===")
            if TORCH_AVAILABLE:
                report.append(f"PyTorch Version: {torch.__version__}")
                report.append(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
                report.append(f"PyTorch CUDA Device Count: {torch.cuda.device_count()}")
                
                # Add detailed GPU information
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    current_device = torch.cuda.current_device()
                    report.append(f"Current CUDA Device: {current_device}")
                    report.append(f"Device Name: {torch.cuda.get_device_name(current_device)}")
                    report.append(f"Device Capability: {torch.cuda.get_device_capability(current_device)}")
                    report.append(f"GPU Memory Allocated: {torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB")
                    report.append(f"GPU Memory Reserved: {torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB")
                    
                    try:
                        report.append(f"PyTorch CUDA Version: {torch.version.cuda}")
                        report.append(f"PyTorch CUDNN Version: {torch.backends.cudnn.version()}")
                        report.append(f"CUDNN Enabled: {torch.backends.cudnn.enabled}")
                    except:
                        report.append("PyTorch CUDA Version: Not available")
            else:
                report.append("PyTorch not installed or not available")
            
            # CUDA info
            report.append("\n=== CUDA System Info ===")
            report.append(f"CUDA Available for Use: {TORCH_AVAILABLE}")
            cuda_device_count = 0
            try:
                cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
                report.append(f"OpenCV CUDA Devices: {cuda_device_count}")
                
                # Try to get CUDA version
                try:
                    import ctypes
                    cuda_found = False
                    
                    # Try different CUDA runtime versions
                    for version in ['124', '123', '121', '120', '118', '117', '116', '115', '114', '110']:
                        try:
                            libcudart = ctypes.CDLL(f'cudart64_{version}.dll')
                            report.append(f"CUDA {version[0]}.{version[1:]} Runtime found")
                            cuda_found = True
                            break
                        except:
                            continue
                            
                    if not cuda_found:
                        report.append("No CUDA Runtime found in system path")
                except:
                    report.append("Error checking CUDA Runtime")
                
                # Try to get NVIDIA driver info
                try:
                    import ctypes
                    nvml = ctypes.CDLL('nvml.dll')
                    report.append("NVIDIA Management Library found")
                except:
                    report.append("NVIDIA Management Library not accessible")
                
            except Exception as e:
                report.append(f"Error getting CUDA info: {str(e)}")
            
            # Run a quick benchmark if PyTorch is available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                report.append("\n=== GPU Quick Benchmark ===")
                try:
                    # Create sample data
                    x = torch.randn(1000, 1000).cuda()
                    y = torch.randn(1000, 1000).cuda()
                    
                    # Warm-up
                    torch.matmul(x, y)
                    torch.cuda.synchronize()
                    
                    # Benchmark
                    start = time.time()
                    for _ in range(10):
                        z = torch.matmul(x, y)
                        torch.cuda.synchronize()
                    end = time.time()
                    
                    report.append(f"Matrix multiplication (1000x1000) time: {(end - start) / 10 * 1000:.2f} ms")
                    
                    # Clean up
                    del x, y, z
                    torch.cuda.empty_cache()
                except Exception as e:
                    report.append(f"Benchmark error: {str(e)}")
            
            # Environment variables
            report.append("\n=== Environment Variables ===")
            cuda_env_vars = ['CUDA_PATH', 'CUDA_HOME', 'PATH', 'NVIDIA_DRIVER_PATH', 'CUDA_DEVICE_ORDER']
            for var in cuda_env_vars:
                if var in os.environ:
                    val = os.environ[var]
                    if var == 'PATH':
                        # Just check if CUDA is in the PATH
                        if 'cuda' in val.lower():
                            report.append(f"{var}: Contains CUDA directory")
                        else:
                            report.append(f"{var}: Does not contain CUDA directory")
                    else:
                        report.append(f"{var}: {val}")
                else:
                    report.append(f"{var}: Not set")

            # Check pip packages
            report.append("\n=== Installed Packages ===")
            try:
                import subprocess
                result = subprocess.run([sys.executable, '-m', 'pip', 'list'], stdout=subprocess.PIPE, text=True)
                packages = result.stdout.splitlines()
                important_packages = [p for p in packages if any(name in p.lower() for name in 
                                     ['opencv', 'torch', 'cuda', 'numpy', 'bettercam', 'pyqt', 'pillow', 'pywin'])]
                
                for package in important_packages:
                    report.append(package)
            except Exception as e:
                report.append(f"Error getting pip packages: {str(e)}")
                
            # Check for conda
            report.append("\n=== Conda Information ===")
            try:
                import subprocess
                result = subprocess.run(['conda', 'info'], stdout=subprocess.PIPE, text=True, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    report.append("Conda is installed")
                    report.append("Recommended: Use conda to install OpenCV with CUDA support")
                else:
                    report.append("Conda not found or not in PATH")
                    report.append("Consider installing Miniconda or Anaconda to easily get OpenCV with CUDA")
            except Exception as e:
                report.append("Conda not found or not in PATH")
                report.append("Consider installing Miniconda or Anaconda to easily get OpenCV with CUDA")
            
            # Current config
            report.append("\n=== Current Configuration ===")
            for key, value in self.config.items():
                report.append(f"{key}: {value}")
            
            # Add recommendations
            report.append("\n=== Recommendations ===")
            recommendations = []
            
            # Check if we have PyTorch with CUDA working
            if TORCH_AVAILABLE and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                recommendations.append("✅ PyTorch with CUDA is properly configured!")
                if not TORCH_AVAILABLE:
                    recommendations.append("Consider implementing PyTorch-based color detection as fallback")
                    recommendations.append("PyTorch provides a reliable GPU acceleration alternative to OpenCV")
            elif TORCH_AVAILABLE and not torch.cuda.is_available():
                recommendations.append("❌ PyTorch is installed but CUDA is not available.")
                recommendations.append("Reinstall PyTorch with CUDA support:")
                recommendations.append("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            else:
                recommendations.append("⚠️ For best performance, install PyTorch with CUDA:")
                recommendations.append("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            
            # Check if OpenCV has CUDA support
            if TORCH_AVAILABLE:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    recommendations.append("✅ OpenCV with CUDA is properly configured!")
                else:
                    recommendations.append("⚠️ OpenCV was compiled with CUDA but no CUDA devices were detected")
                    recommendations.append("Check your GPU drivers or try using PyTorch-based acceleration instead")
            else:
                recommendations.append("❌ OpenCV was not compiled with CUDA support")
                recommendations.append("For proper CUDA support in OpenCV, use Conda:")
                recommendations.append("conda install -c conda-forge opencv opencv-contrib-python")

            # Add all recommendations to the report
            for rec in recommendations:
                report.append(rec)
            
            # Display report in logs
            for line in report:
                self.log(line)
            
            # Save report to file
            report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_report.txt")
            with open(report_path, "w") as f:
                f.write("\n".join(report))
            
            self.log(f"Debug report saved to: {report_path}")
            
        except Exception as e:
            self.log(f"Error generating debug report: {str(e)}")
            traceback.print_exc()

    def update_smart_acceleration(self):
        """Update the smart acceleration setting"""
        smart_accel = self.smart_accel_checkbox.isChecked()
        self.config["smart_acceleration"] = smart_accel
        save_config(self.config)
        
        # Update the triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            
        self.log(f"Smart acceleration {'enabled' if smart_accel else 'disabled'}")
    
    def run_benchmark(self):
        """Run a benchmark to compare CPU vs GPU performance"""
        self.log("Starting performance benchmark...")
        
        # Check if we have a camera instance to grab a frame
        if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread and hasattr(self.trigger_bot_thread, 'triggerbot'):
            triggerbot = self.trigger_bot_thread.triggerbot
            if hasattr(triggerbot, 'camera'):
                # Get a frame from the camera
                frame = triggerbot.camera.get_latest_frame()
                if frame is not None:
                    # Run the benchmark
                    self.log("Running benchmark with current frame...")
                    benchmark_results = benchmark_gpu_vs_cpu(frame)
                    
                    # Log the results
                    self.log(f"Benchmark Results:")
                    self.log(f"CPU time: {benchmark_results['cpu_time']:.2f}ms")
                    
                    if benchmark_results['pytorch_gpu_time'] < float('inf'):
                        self.log(f"PyTorch GPU time: {benchmark_results['pytorch_gpu_time']:.2f}ms")
                        
                    if benchmark_results['cpu_time'] < float('inf'):
                        self.log(f"OpenCV CPU time: {benchmark_results['cpu_time']:.2f}ms")
                    
                    self.log(f"Best method: {benchmark_results['best_method']}")
                    
                    if benchmark_results['ratio'] > 1:
                        self.log(f"GPU is {benchmark_results['ratio']:.2f}x faster than CPU")
                        
                        # Update config if smart acceleration is enabled
                        if self.config.get("smart_acceleration", True):
                            self.config["use_gpu"] = True
                            self.use_gpu_checkbox.setChecked(True)
                            self.log("Smart acceleration: Switched to GPU mode")
                    else:
                        self.log(f"WARNING: GPU is {1/benchmark_results['ratio']:.2f}x SLOWER than CPU")
                        
                        # Update config if smart acceleration is enabled
                        if self.config.get("smart_acceleration", True):
                            self.config["use_gpu"] = False
                            self.use_gpu_checkbox.setChecked(False)
                            self.log("Smart acceleration: Switched to CPU mode")
                    
                    save_config(self.config)
                    
                    # Update the triggerbot if running
                    if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
                        self.trigger_bot_thread.update_config(self.config)
                    
                    return
                else:
                    self.log("Error: Could not get frame from camera")
            else:
                self.log("Error: No camera available")
        else:
            # We don't have a camera, so we can't run a benchmark on a real frame
            self.log("No active camera. Starting Triggerbot to run benchmark...")
            
            # Create a temporary test frame
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_frame[:, :, 1] = 255  # Add some green
            
            # Run the benchmark with the test frame
            benchmark_results = benchmark_gpu_vs_cpu(test_frame)
            
            # Log the results
            self.log(f"Benchmark Results (test frame):")
            self.log(f"CPU time: {benchmark_results['cpu_time']:.2f}ms")
            
            if benchmark_results['pytorch_gpu_time'] < float('inf'):
                self.log(f"PyTorch GPU time: {benchmark_results['pytorch_gpu_time']:.2f}ms")
                
            if benchmark_results['cpu_time'] < float('inf'):
                self.log(f"OpenCV CPU time: {benchmark_results['cpu_time']:.2f}ms")
            
            self.log(f"Best method: {benchmark_results['best_method']}")
            
            if benchmark_results['ratio'] > 1:
                self.log(f"GPU is {benchmark_results['ratio']:.2f}x faster than CPU")
                
                # Update config if smart acceleration is enabled
                if self.config.get("smart_acceleration", True):
                    self.config["use_gpu"] = True
                    self.use_gpu_checkbox.setChecked(True)
                    self.log("Smart acceleration: GPU mode recommended")
            else:
                self.log(f"WARNING: GPU is slower than CPU")
                
                # Update config if smart acceleration is enabled
                if self.config.get("smart_acceleration", True):
                    self.config["use_gpu"] = False
                    self.use_gpu_checkbox.setChecked(False)
                    self.log("Smart acceleration: CPU mode recommended")
            
            save_config(self.config)
    
    def update_gpu_config(self):
        # Update GPU config
        use_gpu = self.use_gpu_checkbox.isChecked()
        self.config["use_gpu"] = use_gpu
        save_config(self.config)
        
        # Update the triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            
        self.log(f"GPU acceleration {'enabled' if use_gpu else 'disabled'}")
        
        # If enabling GPU, suggest running a benchmark
        if use_gpu:
            self.log("Consider running a benchmark to verify GPU performance")

    def update_block_movements(self, state):
        """Update block movements setting and apply immediately to running triggerbot"""
        block_movements = (state == QtCore.Qt.Checked)
        self.config["block_movements"] = block_movements
        save_config(self.config)
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            self.log(f"Block movements {'enabled' if block_movements else 'disabled'} - applied immediately")

    def update_press_duration_min(self, value):
        """Update minimum press duration and apply immediately to running triggerbot"""
        self.config["press_duration_min_s"] = value
        save_config(self.config)
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            self.log(f"Press duration minimum set to {value:.2f}s - applied immediately")

    def update_press_duration_max(self, value):
        """Update maximum press duration and apply immediately to running triggerbot"""
        self.config["press_duration_max_s"] = value
        save_config(self.config)
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            self.log(f"Press duration maximum set to {value:.2f}s - applied immediately")

    def toggle_aim_lock(self, state):
        if state == QtCore.Qt.Checked:
            if not self.aim_lock_controller:
                self.aim_lock_controller = AimLockController(self.config)
            self.aim_lock_controller.update_config(self.config)
            self.aim_lock_controller.start()
        else:
            if self.aim_lock_controller:
                self.aim_lock_controller.stop()
                print('[DEBUG] AimLockController fully stopped')
            # Check if both features are off
            if (not getattr(self, 'trigger_bot_thread', None) or not self.triggerbot_checkbox.isChecked()) and (not self.aim_lock_controller or not self.aim_lock_enabled_checkbox.isChecked()):
                print('[DEBUG] All scanning and status updates stopped (AimLock)')

    def update_performance_mode(self):
        self.config["use_gpu"] = self.gpu_radio.isChecked()
        save_config(self.config)
        # Update active triggerbot and aimlock if running
        if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
        if hasattr(self, 'aim_lock_controller') and self.aim_lock_controller:
            self.aim_lock_controller.update_config(self.config)

    def setup_profiling_tab(self):
        """Setup the Profiling tab UI"""
        layout = QtWidgets.QVBoxLayout(self.profiling_tab)

        # --- Profiling Tab Title ---
        title = QtWidgets.QLabel("<h2>Profile Management</h2>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setAccessibleName("Profile Management Title")
        title.setAccessibleDescription("Section title for profile management features.")
        layout.addWidget(title)

        # --- Active Profile Label ---
        self.active_profile_label = QtWidgets.QLabel("Active Profile: None")
        self.active_profile_label.setStyleSheet("font-weight: bold; color: #32CD32; font-size: 15px;")
        layout.addWidget(self.active_profile_label)

        # --- Shortcut Status Label ---
        self.shortcut_status_label = QtWidgets.QLabel("")
        self.shortcut_status_label.setStyleSheet("color: #FFA500;")
        layout.addWidget(self.shortcut_status_label)

        # --- Auto Mode Status Label ---
        self.auto_status_label = QtWidgets.QLabel("")
        self.auto_status_label.setStyleSheet("color: #FFA500;")
        layout.addWidget(self.auto_status_label)

        # --- Profile Selection Dropdown ---
        profile_select_layout = QtWidgets.QHBoxLayout()
        profile_label = QtWidgets.QLabel("Select Profile:")
        profile_label.setToolTip("Choose a profile to view or edit.")
        self.profile_combo = QtWidgets.QComboBox()
        self.refresh_profiles()
        self.profile_combo.currentIndexChanged.connect(self.on_profile_selected)
        profile_select_layout.addWidget(profile_label)
        profile_select_layout.addWidget(self.profile_combo)
        layout.addLayout(profile_select_layout)

        # --- Auto Mode Toggle ---
        auto_layout = QtWidgets.QHBoxLayout()
        self.auto_checkbox = QtWidgets.QCheckBox("Enable Auto Profile Detection (1/2)")
        self.auto_checkbox.setToolTip("When enabled, pressing 1 or 2 will scan for gun name and auto-load profile.")
        self.auto_checkbox.stateChanged.connect(self.on_auto_mode_toggled)
        auto_layout.addWidget(self.auto_checkbox)
        auto_layout.addStretch()
        layout.addLayout(auto_layout)

        # --- Profile Editor Panel (UI controls, not JSON) ---
        self.profile_editor_group = QtWidgets.QGroupBox("Profile Editor")
        self.profile_editor_group.setToolTip("Edit all settings for the selected profile below.")
        # Use a scroll area for the editor controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        editor_widget = QtWidgets.QWidget()
        self.profile_editor_layout = QtWidgets.QGridLayout(editor_widget)
        scroll_area.setWidget(editor_widget)
        self.profile_editor_group.setLayout(QtWidgets.QVBoxLayout())
        self.profile_editor_group.layout().addWidget(scroll_area)
        layout.addWidget(self.profile_editor_group)
        self.profile_editor_controls = {}
        self.build_profile_editor_controls_grid()

        # --- Save/Reset/Import/Export/Reset-to-Default Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        self.save_profile_btn = QtWidgets.QPushButton("💾 Save Profile")
        self.save_profile_btn.setStyleSheet("font-weight: bold; background-color: #32CD32; color: black;")
        self.save_profile_btn.setToolTip("Save changes to this profile.")
        self.save_profile_btn.clicked.connect(self.save_current_profile)
        self.reset_profile_btn = QtWidgets.QPushButton("↩️ Reset Editor")
        self.reset_profile_btn.setStyleSheet("font-weight: bold; background-color: #FFA500; color: black;")
        self.reset_profile_btn.setToolTip("Reset all fields to the last saved state for this profile.")
        self.reset_profile_btn.clicked.connect(self.reset_profile_editor)
        self.import_profile_btn = QtWidgets.QPushButton("⬆️ Import Profile")
        self.import_profile_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        self.import_profile_btn.setToolTip("Import a profile from a JSON file.")
        self.import_profile_btn.clicked.connect(self.import_profile)
        self.export_profile_btn = QtWidgets.QPushButton("⬇️ Export Profile")
        self.export_profile_btn.setStyleSheet("font-weight: bold; background-color: #607D8B; color: white;")
        self.export_profile_btn.setToolTip("Export the current profile to a JSON file.")
        self.export_profile_btn.clicked.connect(self.export_profile)
        self.reset_default_btn = QtWidgets.QPushButton("🧹 Reset to Default")
        self.reset_default_btn.setStyleSheet("font-weight: bold; background-color: #E91E63; color: white;")
        self.reset_default_btn.setToolTip("Reset this profile to default values.")
        self.reset_default_btn.clicked.connect(self.reset_profile_to_default)
        btn_layout.addWidget(self.save_profile_btn)
        btn_layout.addWidget(self.reset_profile_btn)
        btn_layout.addWidget(self.import_profile_btn)
        btn_layout.addWidget(self.export_profile_btn)
        btn_layout.addWidget(self.reset_default_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.active_profile = None

        # --- Start shortcut listener thread only once ---
        if not hasattr(self, 'shortcut_thread_started'):
            self.profile_shortcut_signal.connect(self.load_profile_by_name)
            self.profile_shortcut_map = {}
            self.shortcut_listener_active = True
            self.shortcut_thread = threading.Thread(target=self.profile_shortcut_listener, daemon=True)
            self.shortcut_thread.start()
            self.shortcut_thread_started = True

        # --- Validation Error Label ---
        self.profile_validation_label = QtWidgets.QLabel("")
        self.profile_validation_label.setStyleSheet("color: #FF3333; font-weight: bold;")
        self.profile_validation_label.setWordWrap(True)
        layout.addWidget(self.profile_validation_label)

    def build_profile_editor_controls_grid(self):
        for i in reversed(range(self.profile_editor_layout.count())):
            widget = self.profile_editor_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.profile_editor_controls.clear()
        config = getattr(self, 'config', {})
        row = 0
        col = 0
        # --- Section: TriggerBot ---
        group1 = QtWidgets.QLabel("<b>TriggerBot Settings</b>")
        group1.setToolTip("Settings for the TriggerBot feature.")
        self.profile_editor_layout.addWidget(group1, row, 0, 1, 2)
        row += 1
        # --- Shortcut Key ---
        shortcut_combo = QtWidgets.QComboBox()
        shortcut_keys = ["None", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4", "Ctrl+5"]
        shortcut_combo.addItems(shortcut_keys)
        shortcut_val = config.get("shortcut_key", "None")
        if shortcut_val in shortcut_keys:
            shortcut_combo.setCurrentText(shortcut_val)
        else:
            shortcut_combo.setCurrentText("None")
        shortcut_combo.setToolTip("Assign a keyboard shortcut to instantly load this profile.")
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Shortcut Key:"), row, 0)
        self.profile_editor_layout.addWidget(shortcut_combo, row, 1)
        self.profile_editor_controls['shortcut_key'] = shortcut_combo
        row += 1
        fov_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        fov_slider.setRange(1, 50)
        fov_slider.setValue(int(config.get("fov", 5.0)))
        fov_slider.setToolTip("Field of View: Controls the detection area size.")
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Detection FOV:"), row, 0)
        self.profile_editor_layout.addWidget(fov_slider, row, 1)
        self.profile_editor_controls['fov'] = fov_slider
        row += 1
        key_combo = QtWidgets.QComboBox()
        key_combo.addItems(list(self.key_map.keys()))
        inv_key_map = {v: k for k, v in self.key_map.items()}
        key = inv_key_map.get(config.get("keybind", 164), "Alt")
        key_combo.setCurrentText(key)
        key_combo.setToolTip("Key to activate TriggerBot.")
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Activation Key:"), row, 0)
        self.profile_editor_layout.addWidget(key_combo, row, 1)
        self.profile_editor_controls['keybind'] = key_combo
        row += 1
        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItems(["hold", "toggle"])
        mode_combo.setCurrentText(config.get("trigger_mode", "hold"))
        mode_combo.setToolTip("TriggerBot activation mode.")
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Trigger Mode:"), row, 0)
        self.profile_editor_layout.addWidget(mode_combo, row, 1)
        self.profile_editor_controls['trigger_mode'] = mode_combo
        row += 1
        min_delay = QtWidgets.QDoubleSpinBox()
        min_delay.setRange(10.0, 500.0)
        min_delay.setValue(config.get("min_shooting_delay_ms", 50.0))
        min_delay.setSingleStep(5.0)
        min_delay.setDecimals(0)
        min_delay.setToolTip("Minimum shooting delay in ms.")
        max_delay = QtWidgets.QDoubleSpinBox()
        max_delay.setRange(10.0, 500.0)
        max_delay.setValue(config.get("max_shooting_delay_ms", 80.0))
        max_delay.setSingleStep(5.0)
        max_delay.setDecimals(0)
        max_delay.setToolTip("Maximum shooting delay in ms.")
        delay_widget = QtWidgets.QWidget()
        delay_layout = QtWidgets.QHBoxLayout(delay_widget)
        delay_layout.setContentsMargins(0, 0, 0, 0)
        delay_layout.addWidget(QtWidgets.QLabel("Min:"))
        delay_layout.addWidget(min_delay)
        delay_layout.addWidget(QtWidgets.QLabel("Max:"))
        delay_layout.addWidget(max_delay)
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Shooting Delay (ms):"), row, 0)
        self.profile_editor_layout.addWidget(delay_widget, row, 1)
        self.profile_editor_controls['min_shooting_delay_ms'] = min_delay
        self.profile_editor_controls['max_shooting_delay_ms'] = max_delay
        row += 1
        fps_spin = QtWidgets.QDoubleSpinBox()
        fps_spin.setRange(30.0, 1000.0)
        fps_spin.setValue(config.get("fps", 200.0))
        fps_spin.setSingleStep(10.0)
        fps_spin.setDecimals(0)
        fps_spin.setToolTip("Target FPS for scanning.")
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Target FPS:"), row, 0)
        self.profile_editor_layout.addWidget(fps_spin, row, 1)
        self.profile_editor_controls['fps'] = fps_spin
        row += 1
        press_duration_toggle = QtWidgets.QCheckBox("Enable Random Press Duration")
        press_duration_toggle.setChecked(config.get("enable_random_press_duration", True))
        press_duration_toggle.setToolTip("Enable randomization of press duration for each shot.")
        self.profile_editor_layout.addWidget(press_duration_toggle, row, 0, 1, 2)
        self.profile_editor_controls['enable_random_press_duration'] = press_duration_toggle
        row += 1
        press_min = QtWidgets.QDoubleSpinBox()
        press_min.setRange(0.01, 1.00)
        press_min.setValue(config.get("press_duration_min_s", 0.01))
        press_min.setSingleStep(0.01)
        press_min.setDecimals(2)
        press_min.setToolTip("Minimum press duration in seconds.")
        press_max = QtWidgets.QDoubleSpinBox()
        press_max.setRange(0.01, 1.00)
        press_max.setValue(config.get("press_duration_max_s", 0.05))
        press_max.setSingleStep(0.01)
        press_max.setDecimals(2)
        press_max.setToolTip("Maximum press duration in seconds.")
        press_widget = QtWidgets.QWidget()
        press_layout = QtWidgets.QHBoxLayout(press_widget)
        press_layout.setContentsMargins(0, 0, 0, 0)
        press_layout.addWidget(QtWidgets.QLabel("Min Duration (s):"))
        press_layout.addWidget(press_min)
        press_layout.addWidget(QtWidgets.QLabel("Max Duration (s):"))
        press_layout.addWidget(press_max)
        self.profile_editor_layout.addWidget(press_widget, row, 0, 1, 2)
        self.profile_editor_controls['press_duration_min_s'] = press_min
        self.profile_editor_controls['press_duration_max_s'] = press_max
        row += 1
        block_movements = QtWidgets.QCheckBox("Block Movement Keys While Shooting")
        block_movements.setChecked(config.get("block_movements", False))
        block_movements.setToolTip("Temporarily release W/A/S/D while shooting.")
        self.profile_editor_layout.addWidget(block_movements, row, 0, 1, 2)
        self.profile_editor_controls['block_movements'] = block_movements
        row += 1
        use_gpu = QtWidgets.QCheckBox("Use GPU Acceleration")
        use_gpu.setChecked(config.get("use_gpu", False))
        use_gpu.setToolTip("Enable GPU acceleration if available.")
        self.profile_editor_layout.addWidget(use_gpu, row, 0, 1, 2)
        self.profile_editor_controls['use_gpu'] = use_gpu
        row += 1
        smart_accel = QtWidgets.QCheckBox("Smart Acceleration")
        smart_accel.setChecked(config.get("smart_acceleration", True))
        smart_accel.setToolTip("Automatically select best acceleration method.")
        self.profile_editor_layout.addWidget(smart_accel, row, 0, 1, 2)
        self.profile_editor_controls['smart_acceleration'] = smart_accel
        row += 1
        test_mode = QtWidgets.QCheckBox("Test Mode")
        test_mode.setChecked(config.get("test_mode", False))
        test_mode.setToolTip("Enable test mode for performance comparison.")
        self.profile_editor_layout.addWidget(test_mode, row, 0, 1, 2)
        self.profile_editor_controls['test_mode'] = test_mode
        row += 1
        theme_combo = QtWidgets.QComboBox()
        theme_combo.addItems(["dark", "light", "custom"])
        theme_combo.setCurrentText(config.get("theme", "custom"))
        theme_combo.setToolTip("UI theme for the app.")
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Theme:"), row, 0)
        self.profile_editor_layout.addWidget(theme_combo, row, 1)
        self.profile_editor_controls['theme'] = theme_combo
        row += 1
        # --- Divider between TriggerBot and Aim Lock ---
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.profile_editor_layout.addWidget(divider, row, 0, 1, 2)
        row += 1
        # --- Section: Aim Lock ---
        group2 = QtWidgets.QLabel("<b>Aim Lock Settings</b>")
        group2.setToolTip("Settings for the Aim Lock feature.")
        self.profile_editor_layout.addWidget(group2, row, 0, 1, 2)
        row += 1
        aim_lock_enabled = QtWidgets.QCheckBox("Enable Aim Lock")
        aim_lock_enabled.setChecked(config.get("aim_lock_enabled", False))
        aim_lock_enabled.setToolTip("Enable or disable Aim Lock.")
        self.profile_editor_layout.addWidget(aim_lock_enabled, row, 0, 1, 2)
        self.profile_editor_controls['aim_lock_enabled'] = aim_lock_enabled
        row += 1
        aim_lock_mode_combo = QtWidgets.QComboBox()
        aim_lock_mode_combo.addItems(["hold", "toggle"])
        aim_lock_mode_combo.setCurrentText(config.get("aim_lock_mode", "hold"))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Mode:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_mode_combo, row, 1)
        self.profile_editor_controls['aim_lock_mode'] = aim_lock_mode_combo
        row += 1
        aim_lock_key_combo = QtWidgets.QComboBox()
        aim_lock_key_combo.addItems(list(self.key_map.keys()))
        key = inv_key_map.get(config.get("aim_lock_keybind", 164), "Alt")
        aim_lock_key_combo.setCurrentText(key)
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Key:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_key_combo, row, 1)
        self.profile_editor_controls['aim_lock_keybind'] = aim_lock_key_combo
        row += 1
        aim_lock_strength = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        aim_lock_strength.setRange(1, 500)
        aim_lock_strength.setValue(config.get("aim_lock_strength", 68))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Strength:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_strength, row, 1)
        self.profile_editor_controls['aim_lock_strength'] = aim_lock_strength
        row += 1
        aim_lock_scan_x = QtWidgets.QSpinBox()
        aim_lock_scan_x.setRange(1, 100)
        aim_lock_scan_x.setValue(config.get("aim_lock_scan_area_x", 16))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Scan Area X:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_scan_x, row, 1)
        self.profile_editor_controls['aim_lock_scan_area_x'] = aim_lock_scan_x
        row += 1
        aim_lock_scan_y = QtWidgets.QSpinBox()
        aim_lock_scan_y.setRange(1, 100)
        aim_lock_scan_y.setValue(config.get("aim_lock_scan_area_y", 10))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Scan Area Y:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_scan_y, row, 1)
        self.profile_editor_controls['aim_lock_scan_area_y'] = aim_lock_scan_y
        row += 1
        aim_lock_debug = QtWidgets.QCheckBox("Aim Lock Debug Mode")
        aim_lock_debug.setChecked(config.get("aim_lock_debug_mode", False))
        self.profile_editor_layout.addWidget(aim_lock_debug, row, 0, 1, 2)
        self.profile_editor_controls['aim_lock_debug_mode'] = aim_lock_debug
        row += 1
        aim_lock_adaptive = QtWidgets.QCheckBox("Aim Lock Adaptive Scan")
        aim_lock_adaptive.setChecked(config.get("aim_lock_adaptive_scan", True))
        self.profile_editor_layout.addWidget(aim_lock_adaptive, row, 0, 1, 2)
        self.profile_editor_controls['aim_lock_adaptive_scan'] = aim_lock_adaptive
        row += 1
        aim_lock_pattern_combo = QtWidgets.QComboBox()
        aim_lock_pattern_combo.addItems(["spiral", "grid"])
        aim_lock_pattern_combo.setCurrentText(config.get("aim_lock_scan_pattern", "spiral"))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Scan Pattern:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_pattern_combo, row, 1)
        self.profile_editor_controls['aim_lock_scan_pattern'] = aim_lock_pattern_combo
        row += 1
        aim_lock_tolerance = QtWidgets.QSpinBox()
        aim_lock_tolerance.setRange(1, 100)
        aim_lock_tolerance.setValue(config.get("aim_lock_tolerance", 45))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Tolerance:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_tolerance, row, 1)
        self.profile_editor_controls['aim_lock_tolerance'] = aim_lock_tolerance
        row += 1
        aim_lock_refresh = QtWidgets.QSpinBox()
        aim_lock_refresh.setRange(30, 1000)
        aim_lock_refresh.setValue(config.get("aim_lock_refresh_rate", 240))
        self.profile_editor_layout.addWidget(QtWidgets.QLabel("Aim Lock Refresh Rate:"), row, 0)
        self.profile_editor_layout.addWidget(aim_lock_refresh, row, 1)
        self.profile_editor_controls['aim_lock_refresh_rate'] = aim_lock_refresh
        # HSV range and aim_lock_target_color are not exposed for now (advanced)

    def on_profile_selected(self, idx):
        if idx < 0 or idx >= len(self.profile_names):
            return
        name = self.profile_names[idx]
        self.load_profile_by_name(name)
        self.build_profile_editor_controls_grid()
        self.update_profile_editor_controls_from_config()

    def load_profile_by_name(self, name):
        # Temporarily disable shortcut listener to prevent recursion/loop
        self.shortcut_listener_active = False
        path = os.path.join('profiles', f'{name}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = f.read()
            try:
                profile_config = json.loads(data)
                # Map old keys to new keys for backward compatibility
                key_map = {
                    "min_shooting_rate": "min_shooting_delay_ms",
                    "max_shooting_rate": "max_shooting_delay_ms",
                    "enable_press_duration": "enable_random_press_duration",
                    "press_duration_min": "press_duration_min_s",
                    "press_duration_max": "press_duration_max_s",
                }
                for old, new in key_map.items():
                    if old in profile_config and new not in profile_config:
                        profile_config[new] = profile_config[old]
                
                # CRITICAL FIX: Keep aim_lock_toggle_key, don't remove it
                inv_key_map = {v: k for k, v in self.key_map.items()}
                # Remove deprecated keys except aim_lock_toggle_key
                for deprecated in ["shooting_rate", "auto_fallback_to_cpu"]:
                    if deprecated in profile_config:
                        del profile_config[deprecated]
                
                # Ensure both aim_lock_toggle_key and aim_lock_keybind are present and consistent
                if 'aim_lock_keybind' in profile_config:
                    key_code = profile_config['aim_lock_keybind']
                    # If toggle key is missing, derive it from keybind
                    if 'aim_lock_toggle_key' not in profile_config:
                        key_str = inv_key_map.get(key_code, 'Alt')
                        profile_config['aim_lock_toggle_key'] = key_str.lower()
                    # Force lowercase for keyboard detection
                    elif profile_config['aim_lock_toggle_key'] is not None:
                        profile_config['aim_lock_toggle_key'] = profile_config['aim_lock_toggle_key'].lower()
                
                # Ensure aim_lock_target_color is always an array
                if "aim_lock_target_color" in profile_config and not isinstance(profile_config["aim_lock_target_color"], list):
                    profile_config["aim_lock_target_color"] = [30, 255, 255]
                self.config = profile_config.copy()
                self.active_profile = name
                self.active_profile_label.setText(f"Active Profile: {name}")
                self.apply_profile_config()
                self.update_profile_editor_controls_from_config()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Profile Load Failed", f"Failed to load profile: {e}")
        else:
            self.active_profile = None
            self.active_profile_label.setText("Active Profile: None")
        # Re-enable shortcut listener after profile load
        self.shortcut_listener_active = True

    def update_profile_editor_controls_from_config(self):
        config = getattr(self, 'config', {})
        # Update all controls to match config
        self.profile_editor_controls['fov'].setValue(int(config.get("fov", 5.0)))
        inv_key_map = {v: k for k, v in self.key_map.items()}
        key = inv_key_map.get(config.get("keybind", 164), "Alt")
        self.profile_editor_controls['keybind'].setCurrentText(key)
        self.profile_editor_controls['trigger_mode'].setCurrentText(config.get("trigger_mode", "hold"))
        self.profile_editor_controls['min_shooting_delay_ms'].setValue(config.get("min_shooting_delay_ms", 50.0))
        self.profile_editor_controls['max_shooting_delay_ms'].setValue(config.get("max_shooting_delay_ms", 80.0))
        self.profile_editor_controls['fps'].setValue(config.get("fps", 200.0))
        self.profile_editor_controls['enable_random_press_duration'].setChecked(config.get("enable_random_press_duration", True))
        self.profile_editor_controls['press_duration_min_s'].setValue(config.get("press_duration_min_s", 0.01))
        self.profile_editor_controls['press_duration_max_s'].setValue(config.get("press_duration_max_s", 0.05))
        self.profile_editor_controls['block_movements'].setChecked(config.get("block_movements", False))
        self.profile_editor_controls['use_gpu'].setChecked(config.get("use_gpu", False))
        self.profile_editor_controls['smart_acceleration'].setChecked(config.get("smart_acceleration", True))
        self.profile_editor_controls['test_mode'].setChecked(config.get("test_mode", False))
        self.profile_editor_controls['theme'].setCurrentText(config.get("theme", "custom"))
        self.profile_editor_controls['aim_lock_enabled'].setChecked(config.get("aim_lock_enabled", False))
        self.profile_editor_controls['aim_lock_mode'].setCurrentText(config.get("aim_lock_mode", "hold"))
        key = inv_key_map.get(config.get("aim_lock_keybind", 164), "Alt")
        self.profile_editor_controls['aim_lock_keybind'].setCurrentText(key)
        self.profile_editor_controls['aim_lock_strength'].setValue(config.get("aim_lock_strength", 68))
        self.profile_editor_controls['aim_lock_scan_area_x'].setValue(config.get("aim_lock_scan_area_x", 16))
        self.profile_editor_controls['aim_lock_scan_area_y'].setValue(config.get("aim_lock_scan_area_y", 10))
        self.profile_editor_controls['aim_lock_debug_mode'].setChecked(config.get("aim_lock_debug_mode", False))
        self.profile_editor_controls['aim_lock_adaptive_scan'].setChecked(config.get("aim_lock_adaptive_scan", True))
        self.profile_editor_controls['aim_lock_scan_pattern'].setCurrentText(config.get("aim_lock_scan_pattern", "spiral"))
        self.profile_editor_controls['aim_lock_tolerance'].setValue(config.get("aim_lock_tolerance", 45))
        self.profile_editor_controls['aim_lock_refresh_rate'].setValue(config.get("aim_lock_refresh_rate", 240))
        self.profile_editor_controls['shortcut_key'].setCurrentText(config.get("shortcut_key", "None"))

    def save_current_profile(self):
        name = self.profile_combo.currentText()
        path = os.path.join('profiles', f'{name}.json')
        # Gather values from controls (only new keys)
        config = {}
        config['fov'] = self.profile_editor_controls['fov'].value()
        key = self.profile_editor_controls['keybind'].currentText()
        config['keybind'] = self.key_map.get(key, 164)
        config['trigger_mode'] = self.profile_editor_controls['trigger_mode'].currentText()
        config['min_shooting_delay_ms'] = self.profile_editor_controls['min_shooting_delay_ms'].value()
        config['max_shooting_delay_ms'] = self.profile_editor_controls['max_shooting_delay_ms'].value()
        config['fps'] = self.profile_editor_controls['fps'].value()
        config['enable_random_press_duration'] = self.profile_editor_controls['enable_random_press_duration'].isChecked()
        config['press_duration_min_s'] = self.profile_editor_controls['press_duration_min_s'].value()
        config['press_duration_max_s'] = self.profile_editor_controls['press_duration_max_s'].value()
        config['block_movements'] = self.profile_editor_controls['block_movements'].isChecked()
        config['use_gpu'] = self.profile_editor_controls['use_gpu'].isChecked()
        config['smart_acceleration'] = self.profile_editor_controls['smart_acceleration'].isChecked()
        config['test_mode'] = self.profile_editor_controls['test_mode'].isChecked()
        config['theme'] = self.profile_editor_controls['theme'].currentText()
        config['aim_lock_enabled'] = self.profile_editor_controls['aim_lock_enabled'].isChecked()
        config['aim_lock_mode'] = self.profile_editor_controls['aim_lock_mode'].currentText()
        aim_lock_key = self.profile_editor_controls['aim_lock_keybind'].currentText()
        config['aim_lock_keybind'] = self.key_map.get(aim_lock_key, 164)
        config['aim_lock_toggle_key'] = aim_lock_key  # Ensure this is saved for persistence
        config['aim_lock_strength'] = self.profile_editor_controls['aim_lock_strength'].value()
        config['aim_lock_scan_area_x'] = self.profile_editor_controls['aim_lock_scan_area_x'].value()
        config['aim_lock_scan_area_y'] = self.profile_editor_controls['aim_lock_scan_area_y'].value()
        config['aim_lock_debug_mode'] = self.profile_editor_controls['aim_lock_debug_mode'].isChecked()
        config['aim_lock_adaptive_scan'] = self.profile_editor_controls['aim_lock_adaptive_scan'].isChecked()
        config['aim_lock_scan_pattern'] = self.profile_editor_controls['aim_lock_scan_pattern'].currentText()
        config['aim_lock_tolerance'] = self.profile_editor_controls['aim_lock_tolerance'].value()
        config['aim_lock_refresh_rate'] = self.profile_editor_controls['aim_lock_refresh_rate'].value()
        config['shortcut_key'] = self.profile_editor_controls['shortcut_key'].currentText()
        # Validate before saving
        all_profiles = {}
        for fname in os.listdir('profiles'):
            if fname.endswith('.json'):
                n = os.path.splitext(fname)[0]
                with open(os.path.join('profiles', fname), 'r') as pf:
                    try:
                        all_profiles[n] = json.load(pf)
                    except Exception:
                        pass
        is_valid, err = validate_profile(config, all_profiles, name)
        if not is_valid:
            self.profile_validation_label.setText(f"Validation Error: {err}")
            QtWidgets.QMessageBox.warning(self, "Save Failed", f"Profile validation failed: {err}")
            return
        else:
            self.profile_validation_label.setText("")
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
            QtWidgets.QMessageBox.information(self, "Profile Saved", f"Profile '{name}' saved successfully.")
            self.config = config.copy()
            self.apply_profile_config()
            current_profile = name
            self.refresh_profiles()
            index = self.profile_combo.findText(current_profile)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Failed", f"Failed to save profile: {e}")

    def reset_profile_editor(self):
        self.load_profile_by_name(self.profile_combo.currentText())
        self.update_profile_editor_controls_from_config()

    def on_auto_mode_toggled(self, state):
        if state == QtCore.Qt.Checked:
            self.start_auto_profile_detection()
        else:
            self.stop_auto_profile_detection()

    def start_auto_profile_detection(self):
        if hasattr(self, 'auto_mode_active') and self.auto_mode_active:
            # Already running
            return
            
        self.auto_status_label.setText("Auto mode enabled. Waiting for '1' or '2' key press...")
        self.auto_status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.log("Auto profile detection mode enabled")
        self.auto_mode_active = True
        self.auto_thread = threading.Thread(target=self.auto_profile_detection_loop, daemon=True)
        self.auto_thread.start()
        debug_log("Auto profile detection thread started")

    def stop_auto_profile_detection(self):
        if hasattr(self, 'auto_mode_active'):
            self.auto_mode_active = False
            
        self.auto_status_label.setText("Auto mode disabled")
        self.auto_status_label.setStyleSheet("color: #FFA500;")
        self.log("Auto profile detection mode disabled")
        
        # Give thread time to exit
        if hasattr(self, 'auto_thread') and self.auto_thread and self.auto_thread.is_alive():
            debug_log("Waiting for auto profile detection thread to exit")
            time.sleep(0.5)  # Give thread time to exit gracefully

    def auto_profile_detection_loop(self):
        import win32api, win32con
        import time
        
        # Load gun names
        with open('all_guns.json', 'r') as f:
            guns = json.load(f)
        all_guns = [g.lower() for cat in guns.values() for g in cat]
        
        # Load OCR model once
        try:
            reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        except Exception:
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Key codes for number keys 1 and 2
        KEY_1 = 0x31  # Virtual key code for '1'
        KEY_2 = 0x32  # Virtual key code for '2'
        
        # Track previous key states to detect key presses
        last_key_states = {
            KEY_1: False,
            KEY_2: False
        }
        last_press = 0
        
        debug_log("Auto profile detection thread started")
        self.log("Auto profile detection enabled - waiting for '1' or '2' key press")
        
        while self.auto_mode_active:
            current_time = time.time()
            
            # Get current key states
            key1_state = win32api.GetAsyncKeyState(KEY_1) < 0
            key2_state = win32api.GetAsyncKeyState(KEY_2) < 0
            
            # Check if key 1 was pressed (transition from not pressed to pressed)
            if key1_state and not last_key_states[KEY_1]:
                debug_log("Detected key '1' press")
                if current_time - last_press > 1.0:
                    last_press = current_time
                    self.auto_status_label.setText("Key '1' pressed - Scanning for weapon name...")
                    self.log("Key '1' pressed - starting weapon scan")
                    QtCore.QMetaObject.invokeMethod(self, "scan_and_load_profile", QtCore.Qt.QueuedConnection)
                    time.sleep(0.3)  # Prevent multiple rapid scans
            
            # Check if key 2 was pressed (transition from not pressed to pressed)
            if key2_state and not last_key_states[KEY_2]:
                debug_log("Detected key '2' press")
                if current_time - last_press > 1.0:
                    last_press = current_time
                    self.auto_status_label.setText("Key '2' pressed - Scanning for weapon name...")
                    self.log("Key '2' pressed - starting weapon scan")
                    QtCore.QMetaObject.invokeMethod(self, "scan_and_load_profile", QtCore.Qt.QueuedConnection)
                    time.sleep(0.3)  # Prevent multiple rapid scans
            
            # Update last key states
            last_key_states[KEY_1] = key1_state
            last_key_states[KEY_2] = key2_state
            
            # Add a small sleep to avoid high CPU usage
            time.sleep(0.05)

    @QtCore.pyqtSlot()
    def scan_and_load_profile(self):
        # Capture region and run OCR
        debug_log("Starting scan_and_load_profile")
        self.auto_status_label.setText("Capturing screen region...")
        
        try:
            img = self.capture_bottom_right_ocr_region()
            if img is None or img.size == 0:
                self.auto_status_label.setText("Error: Failed to capture screen region")
                self.log("Auto Mode Error: Failed to capture screen region")
                debug_log("Failed to capture screen region - image is empty")
                return
                
            debug_log(f"Captured image with shape: {img.shape}")
            self.auto_status_label.setText("Processing image...")
            gray = self.preprocess_ocr_img(img)
            
            # Get or create OCR reader
            reader = getattr(self, '_easyocr_reader', None)
            if reader is None:
                self.auto_status_label.setText("Initializing OCR engine...")
                try:
                    debug_log("Initializing EasyOCR with GPU")
                    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                except Exception as e:
                    debug_log(f"Failed to initialize GPU OCR: {e}, falling back to CPU")
                    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self._easyocr_reader = reader
            
            # Perform OCR
            self.auto_status_label.setText("Running OCR on captured region...")
            debug_log("Starting OCR text detection")
            results = reader.readtext(gray)
            debug_log(f"OCR results: {results}")
            
            detected_texts = [text.lower() for (_, text, conf) in results]
            debug_log(f"Detected texts: {detected_texts}")
            
            # Load gun names
            with open('all_guns.json', 'r') as f:
                guns = json.load(f)
            all_guns = [g.lower() for cat in guns.values() for g in cat]
            
            # Check for matches
            for text in detected_texts:
                for gun in all_guns:
                    if gun in text:
                        self.auto_status_label.setText(f"Detected: {gun.title()} (auto-loading profile)")
                        self.log(f"[Auto Mode] Detected and loaded profile: {gun.title()}")
                        debug_log(f"Found matching gun: {gun} in text: {text}")
                        
                        # Visually highlight the loaded profile in the dropdown
                        idx = self.profile_combo.findText(gun.title())
                        if idx >= 0:
                            self.profile_combo.setCurrentIndex(idx)
                            self.profile_combo.setStyleSheet("QComboBox { background-color: #FFF176; font-weight: bold; }")
                            QtCore.QTimer.singleShot(1500, lambda: self.profile_combo.setStyleSheet(""))
                        else:
                            debug_log(f"Profile not found in combo box: {gun.title()}")
                        
                        self.load_profile_by_name(gun.title())
                        return
            
            # No match found
            debug_log("No matching weapon detected in OCR results")
            self.auto_status_label.setText("No matching weapon detected. Ready for next key press.")
            self.log("[Auto Mode] No matching weapon detected.")
            
        except Exception as e:
            debug_log(f"Error in scan_and_load_profile: {e}")
            traceback.print_exc()
            self.auto_status_label.setText(f"OCR error: {str(e)[:50]}...")
            self.log(f"[Auto Mode] OCR error: {e}")

    def capture_bottom_right_ocr_region(self):
        # Mimic weapon_detections.py logic
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            width, height = monitor['width'], monitor['height']
            region = {
                'left': int(width * 0.9),
                'top': int(height * 0.7),
                'width': int(width * 0.1),
                'height': int(height * 0.3)
            }
            img = np.asarray(sct.grab(region))[..., :3]
            h, w = img.shape[:2]
            crop_left = min(20, w-1)
            crop_right = max(w - 89, crop_left+1)
            crop_bottom = max(h - 185, 1)
            cropped = img[0:crop_bottom, crop_left:crop_right]
            return cropped

    def preprocess_ocr_img(self, img):
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def refresh_profiles(self):
        self.profile_names = []
        if not os.path.exists('profiles'):
            os.makedirs('profiles')
        for fname in sorted(os.listdir('profiles')):
            if fname.endswith('.json'):
                profile_name = os.path.splitext(fname)[0]
                self.profile_names.append(profile_name)
        if hasattr(self, 'profile_combo'):
            self.profile_combo.clear()
            self.profile_combo.addItems(self.profile_names)

    def get_current_shortcut_map(self):
        shortcut_map = {}
        if not os.path.exists('profiles'):
            return shortcut_map
        for fname in sorted(os.listdir('profiles')):
            if fname.endswith('.json'):
                profile_name = os.path.splitext(fname)[0]
                try:
                    with open(os.path.join('profiles', fname), 'r') as f:
                        data = json.load(f)
                        shortcut = data.get('shortcut_key', None)
                        if shortcut and shortcut != "None":
                            shortcut_map[shortcut] = profile_name
                except Exception:
                    pass
        return shortcut_map

    def profile_shortcut_listener(self):
        import win32api
        import win32con
        import time
        print('[DEBUG] Shortcut listener thread started')
        key_map = {
            "F6": win32con.VK_F6,
            "F7": win32con.VK_F7,
            "F8": win32con.VK_F8,
            "F9": win32con.VK_F9,
            "F10": win32con.VK_F10,
            "F11": win32con.VK_F11,
            "F12": win32con.VK_F12,
        }
        ctrl_map = {
            "Ctrl+1": 0x31,  # 1
            "Ctrl+2": 0x32,
            "Ctrl+3": 0x33,
            "Ctrl+4": 0x34,
            "Ctrl+5": 0x35,
        }
        last_key_state = {}
        last_map_print = 0
        last_shortcut_map = None
        while True:
            now = time.time()
            shortcut_map = self.get_current_shortcut_map()
            # Only print if changed or every 10 seconds
            if shortcut_map != last_shortcut_map or now - last_map_print > 10:
                print(f'[DEBUG] Current shortcut map: {shortcut_map}')
                last_map_print = now
                last_shortcut_map = shortcut_map.copy()
            for shortcut, profile in shortcut_map.items():
                if shortcut in key_map:
                    vk = key_map[shortcut]
                    pressed = win32api.GetAsyncKeyState(vk) < 0
                    prev = last_key_state.get(shortcut, False)
                    if pressed and not prev:
                        print(f"[DEBUG] Shortcut pressed: {shortcut} -> {profile}")
                        self.profile_shortcut_signal.emit(profile)
                        self.shortcut_status_label.setText(f"Shortcut: Loaded {profile}")
                    last_key_state[shortcut] = pressed
                elif shortcut in ctrl_map:
                    vk = ctrl_map[shortcut]
                    ctrl_pressed = win32api.GetAsyncKeyState(win32con.VK_CONTROL) < 0
                    key_pressed = win32api.GetAsyncKeyState(vk) < 0
                    pressed = ctrl_pressed and key_pressed
                    prev = last_key_state.get(shortcut, False)
                    if pressed and not prev:
                        print(f"[DEBUG] Shortcut pressed: {shortcut} -> {profile}")
                        self.profile_shortcut_signal.emit(profile)
                        self.shortcut_status_label.setText(f"Shortcut: Loaded {profile}")
                    last_key_state[shortcut] = pressed
            time.sleep(0.05)

    def apply_profile_config(self):
        # Apply config to running modules (TriggerBot, AimLock, etc.)
        if hasattr(self, 'trigger_bot_thread') and self.trigger_bot_thread and hasattr(self.trigger_bot_thread, 'update_config') and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
        
        # CRITICAL FIX: Force aimlock controller reinitialization for key persistence
        print('[DEBUG] Applying profile: aim_lock_toggle_key =', self.config.get('aim_lock_toggle_key'), 'aim_lock_keybind =', self.config.get('aim_lock_keybind'))
        if hasattr(self, 'aim_lock_controller') and self.aim_lock_controller:
            # Fully reinitialize controller to ensure it picks up the new key
            self.aim_lock_controller.stop()
            del self.aim_lock_controller
            self.aim_lock_controller = AimLockController(self.config)
            if self.config.get('aim_lock_enabled', False):
                self.aim_lock_controller.start()
        
        if hasattr(self, 'log'):
            self.log(f"Profile '{self.active_profile}' loaded and applied.")

    # Utility: Batch-migrate all profiles in 'profiles' directory to new schema
    def migrate_all_profiles_to_new_schema(self):
        profiles_dir = 'profiles'
        key_map = {
            "min_shooting_rate": "min_shooting_delay_ms",
            "max_shooting_rate": "max_shooting_delay_ms",
            "enable_press_duration": "enable_random_press_duration",
            "press_duration_min": "press_duration_min_s",
            "press_duration_max": "press_duration_max_s",
        }
        deprecated = ["shooting_rate", "auto_fallback_to_cpu", "aim_lock_toggle_key"]
        for fname in os.listdir(profiles_dir):
            if fname.endswith('.json'):
                path = os.path.join(profiles_dir, fname)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    # Map old keys
                    for old, new in key_map.items():
                        if old in data and new not in data:
                            data[new] = data[old]
                    # Remove deprecated
                    for dep in deprecated:
                        if dep in data:
                            del data[dep]
                    # Ensure aim_lock_target_color is always an array
                    if "aim_lock_target_color" in data and not isinstance(data["aim_lock_target_color"], list):
                        data["aim_lock_target_color"] = [30, 255, 255]
                    # Only keep new keys
                    allowed_keys = [
                        'fov', 'keybind', 'trigger_mode', 'min_shooting_delay_ms', 'max_shooting_delay_ms', 'fps',
                        'hsv_range', 'enable_random_press_duration', 'press_duration_min_s', 'press_duration_max_s',
                        'block_movements', 'use_gpu', 'smart_acceleration', 'test_mode', 'theme',
                        'aim_lock_enabled', 'aim_lock_mode', 'aim_lock_keybind', 'aim_lock_strength',
                        'aim_lock_scan_area_x', 'aim_lock_scan_area_y', 'aim_lock_debug_mode',
                        'aim_lock_adaptive_scan', 'aim_lock_scan_pattern', 'aim_lock_tolerance',
                        'aim_lock_refresh_rate', 'shortcut_key', 'aim_lock_target_color'
                    ]
                    data = {k: v for k, v in data.items() if k in allowed_keys}
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"Migrated profile: {fname}")
                except Exception as e:
                    print(f"Failed to migrate {fname}: {e}")

    def import_profile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Profile", "", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                imported = json.load(f)
            # Validate imported profile
            all_profiles = {}
            for fname in os.listdir('profiles'):
                if fname.endswith('.json'):
                    n = os.path.splitext(fname)[0]
                    with open(os.path.join('profiles', fname), 'r') as pf:
                        try:
                            all_profiles[n] = json.load(pf)
                        except Exception:
                            pass
            is_valid, err = validate_profile(imported, all_profiles)
            if not is_valid:
                QtWidgets.QMessageBox.warning(self, "Import Failed", f"Profile validation failed: {err}")
                return
            # Ask for profile name
            name, ok = QtWidgets.QInputDialog.getText(self, "Profile Name", "Enter a name for the imported profile:")
            if not ok or not name.strip():
                return
            name = name.strip()
            if name in all_profiles:
                QtWidgets.QMessageBox.warning(self, "Import Failed", f"A profile named '{name}' already exists.")
                return
            # Save imported profile
            with open(os.path.join('profiles', f'{name}.json'), 'w') as f:
                json.dump(imported, f, indent=4)
            QtWidgets.QMessageBox.information(self, "Import Successful", f"Profile '{name}' imported successfully.")
            self.refresh_profiles()
            index = self.profile_combo.findText(name)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import Failed", f"Failed to import profile: {e}")

    def export_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Export Failed", "No profile selected to export.")
            return
        path = os.path.join('profiles', f'{name}.json')
        if not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "Export Failed", f"Profile file '{name}.json' not found.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Profile", f"{name}.json", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(path, 'r') as f:
                data = f.read()
            with open(file_path, 'w') as f:
                f.write(data)
            QtWidgets.QMessageBox.information(self, "Export Successful", f"Profile '{name}' exported successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Failed", f"Failed to export profile: {e}")

    def reset_profile_to_default(self):
        name = self.profile_combo.currentText()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Reset Failed", "No profile selected to reset.")
            return
        # Confirm with user
        reply = QtWidgets.QMessageBox.question(self, "Reset to Default", f"Are you sure you want to reset profile '{name}' to default values? This cannot be undone.", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            default_profile = load_config()  # Get default values
            with open(os.path.join('profiles', f'{name}.json'), 'w') as f:
                json.dump(default_profile, f, indent=4)
            QtWidgets.QMessageBox.information(self, "Reset Successful", f"Profile '{name}' has been reset to default values.")
            self.load_profile_by_name(name)
            self.update_profile_editor_controls_from_config()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reset Failed", f"Failed to reset profile: {e}")

def migrate_aimlock_key_fields():
    # Migrate config.json
    config_path = 'config.json'
    key_map = {
        "Alt": 164, "Shift": 160, "Caps Lock": 20, "Tab": 9, "X": 0x58, "C": 0x43, "Z": 0x5A, "V": 0x56,
        "Mouse Right": 0x02, "Mouse 3": 0x04, "Mouse 4": 0x05, "Mouse 5": 0x06
    }
    inv_key_map = {v: k for k, v in key_map.items()}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data = json.load(f)
        if 'aim_lock_keybind' in data and 'aim_lock_toggle_key' not in data:
            key_code = data['aim_lock_keybind']
            key_str = inv_key_map.get(key_code, 'Alt')
            data['aim_lock_toggle_key'] = key_str
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=4)
    # Migrate all profiles
    profiles_dir = 'profiles'
    if os.path.exists(profiles_dir):
        for fname in os.listdir(profiles_dir):
            if fname.endswith('.json'):
                path = os.path.join(profiles_dir, fname)
                with open(path, 'r') as f:
                    pdata = json.load(f)
                if 'aim_lock_keybind' in pdata and 'aim_lock_toggle_key' not in pdata:
                    key_code = pdata['aim_lock_keybind']
                    key_str = inv_key_map.get(key_code, 'Alt')
                    pdata['aim_lock_toggle_key'] = key_str
                    with open(path, 'w') as f:
                        json.dump(pdata, f, indent=4)

# Call migration at startup
migrate_aimlock_key_fields()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
