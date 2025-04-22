import cv2
import numpy as np
import time
import random
import threading
import traceback
import ctypes
import win32api
from multiprocessing import Process, Queue
import queue
from PyQt5 import QtCore
import torch
from shared_scanner import SharedScanner, ScanningManager

# Import PyQt5 for QThread
try:
    from PyQt5 import QtCore
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    
# Check and import PyTorch if available
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available() 
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# This will be shared across processes
shutdown_event = threading.Event()

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
    print(line)

def debug_log(message):
    """
    Log a message with timestamp.
    This is a utility function used throughout the module.
    """
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

def simulate_shoot(q, config):
    keybd_event = ctypes.windll.user32.keybd_event
    while not shutdown_event.is_set():
        try:
            try:
                signal_value = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if signal_value == "UpdateConfig":
                try:
                    updated_config = q.get(timeout=0.1)
                    # Backward compatibility for multiprocessing
                    key_map = {
                        "min_shooting_rate": "min_shooting_delay_ms",
                        "max_shooting_rate": "max_shooting_delay_ms",
                        "enable_press_duration": "enable_random_press_duration",
                        "press_duration_min": "press_duration_min_s",
                        "press_duration_max": "press_duration_max_s",
                    }
                    for old, new in key_map.items():
                        if old in updated_config and new not in updated_config:
                            updated_config[new] = updated_config[old]
                    config.update(updated_config)
                    debug_log("Shooting process config updated: " +
                             f"block_movements={config.get('block_movements', False)}, " +
                             f"enable_random_press_duration={config.get('enable_random_press_duration', True)}, " +
                             f"press_duration_min_s={config.get('press_duration_min_s', 0.01)}, " +
                             f"press_duration_max_s={config.get('press_duration_max_s', 0.03)}")
                    continue
                except Exception as e:
                    debug_log(f"Error updating config in shoot process: {type(e).__name__}: {str(e)}")
                    continue
            if signal_value == "Shoot":
                if config.get("enable_random_press_duration", True):
                    press_duration_min = config.get("press_duration_min_s", 0.01)
                    press_duration_max = config.get("press_duration_max_s", 0.05)
                    press_duration = random.uniform(press_duration_min, press_duration_max)
                else:
                    press_duration = 0
                if config.get("block_movements", False):
                    for key in [0x57, 0x41, 0x53, 0x44]:
                        if win32api.GetAsyncKeyState(key) < 0:
                            keybd_event(key, 0, 2, 0)
                keybd_event(0x01, 0, 0, 0)
                time.sleep(press_duration)
                keybd_event(0x01, 0, 2, 0)
        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            debug_log(f"Error in shoot process: {error_info}")
            traceback.print_exc()
            continue

def fast_benchmark(frame, iterations=10):
    """
    Quickly benchmark GPU vs CPU performance on the input frame.
    This is used for runtime performance optimization.
    
    Args:
        frame: Input frame (numpy array)
        iterations: Number of test iterations
        
    Returns:
        dict with benchmark results
    """
    # Basic benchmark implementation
    results = {
        "cpu_time": 0.0001,
        "gpu_time": float('inf'),
        "ratio": 0,
        "best_method": "CPU"
    }
    
    # CPU test
    cpu_start = time.time()
    for _ in range(iterations):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([30,100,100], dtype=np.uint8), 
                                 np.array([60,255,255], dtype=np.uint8))
        detected = np.any(mask)
    cpu_time = (time.time() - cpu_start) * 1000 / iterations
    results["cpu_time"] = max(cpu_time, 0.0001)
    
    # GPU test if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            # Warmup
            torch.cuda.empty_cache()
            test_tensor = torch.from_numpy(frame).cuda().float() / 255.0
            torch.cuda.synchronize()
            del test_tensor
            
            gpu_start = time.time()
            for _ in range(iterations):
                with torch.amp.autocast('cuda', enabled=True):
                    frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                    r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]
                    max_val, _ = torch.max(frame_tensor, dim=0)
                    min_val, _ = torch.min(frame_tensor, dim=0)
                    diff = max_val - min_val
                    torch.cuda.synchronize()
            gpu_time = (time.time() - gpu_start) * 1000 / iterations
            results["gpu_time"] = max(gpu_time, 0.0001)
        except Exception as e:
            debug_log(f"GPU benchmark error: {e}")
    
    # Calculate performance ratio and select best method
    if results["gpu_time"] < float('inf'):
        results["ratio"] = results["cpu_time"] / results["gpu_time"]
        results["best_method"] = "GPU" if results["ratio"] > 1.0 else "CPU"
    
    return results

class Triggerbot:
    def __init__(self, q, settings, fov, hsv_range, min_shooting_delay_ms, max_shooting_delay_ms, fps):
        self.queue = q
        self.settings = settings
        self.min_shooting_delay_ms = settings.get("min_shooting_delay_ms", 50.0) / 1000.0
        self.max_shooting_delay_ms = settings.get("max_shooting_delay_ms", 80.0) / 1000.0
        self.fps = int(fps)
        self.fov = int(fov)
        self.hsv_range = hsv_range
        self.frames_processed = 0
        self.last_shot_time = 0
        self.check_region = None
        self.stop_flag = False
        self.paused = False
        self.use_gpu = settings.get("use_gpu", False) and CUDA_AVAILABLE
        self.cmin = hsv_range[0]
        self.cmax = hsv_range[1]
        self.update_config_lock = threading.Lock()
        user32 = ctypes.windll.user32
        self.WIDTH, self.HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.update_check_region()
        debug_log(f"Triggerbot initialized with GPU={self.use_gpu}")
        self.manager = ScanningManager.get_instance(fps=self.fps)
        self.region_name = f"triggerbot_{id(self)}"
        self.last_detected = None
        self.last_log_time = 0
        self.color_detected_count = 0
        self.active = False
        def detection_callback(detected, frame, region_name):
            with self.update_config_lock:
                if not self.active or self.stop_flag:
                    return
                if detected != self.last_detected:
                    log_event(f"Target {'acquired' if detected else 'lost'}!", "INFO", context=f"DEVICE={'GPU' if self.use_gpu else 'CPU'}", dedup_key="detection")
                    self.last_detected = detected
                if detected:
                    self.color_detected_count += 1
                    current_time = time.time()
                    random_delay = random.uniform(self.min_shooting_delay_ms, self.max_shooting_delay_ms)
                    if current_time - self.last_shot_time >= random_delay:
                        log_event(f"Shoot signal sent (delay: {random_delay*1000:.0f}ms)", "DEBUG", context="TRIGGERBOT", dedup_key="shoot")
                        self.queue.put("Shoot")
                        self.last_shot_time = current_time
        self._on_detection_result = detection_callback
        scan_params = {
            'mode': 'gpu' if self.use_gpu else 'cpu',
            'color_range': (self.cmin, self.cmax),
            'tolerance': 45,
            'color_space': 'hsv',
        }
        self.manager.register_subscriber(
            self.region_name,
            self.check_region,
            scan_params,
            self._on_detection_result
        )
        self.manager.start()

    def update_check_region(self):
        try:
            center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
            fov_size = max(4, self.fov)
            left = center_x - fov_size
            top = center_y - fov_size
            right = center_x + fov_size
            bottom = center_y + fov_size
            left = max(0, left)
            top = max(0, top)
            right = min(self.WIDTH, right)
            bottom = min(self.HEIGHT, bottom)
            self.check_region = {
                'left': left,
                'top': top,
                'width': right - left,
                'height': bottom - top
            }
            debug_log(f"Updated check region to {self.check_region}")
        except Exception as e:
            debug_log(f"Error updating check region: {e}")
            traceback.print_exc()

    def update_hsv_range(self, hsv_range):
        self.hsv_range = hsv_range
        self.cmin = hsv_range[0]
        self.cmax = hsv_range[1]
        debug_log(f"Updated HSV range to {hsv_range}")
        self.manager.update_region(self.region_name, scan_params={
            'mode': 'gpu' if self.use_gpu else 'cpu',
            'color_range': (self.cmin, self.cmax),
            'tolerance': 45,
            'color_space': 'hsv',
        })

    def update_config(self, settings):
        with self.update_config_lock:
            self.settings = settings
            self.fov = int(settings.get("fov", 5.0))
            self.min_shooting_delay_ms = settings.get("min_shooting_delay_ms", 50.0) / 1000.0
            self.max_shooting_delay_ms = settings.get("max_shooting_delay_ms", 80.0) / 1000.0
            self.use_gpu = settings.get("use_gpu", False) and CUDA_AVAILABLE
            self.update_hsv_range(settings.get("hsv_range", [[30,125,150],[30,255,255]]))

    def run(self):
        try:
            log_event("TriggerBot thread started (ScanningManager)", "INFO", context=f"MODE={self.settings.get('trigger_mode','hold').upper()} DEVICE={'GPU' if self.use_gpu else 'CPU'}")
            self.last_detected = None
            self.last_log_time = 0
            self.color_detected_count = 0
            while not self.stop_flag and not shutdown_event.is_set():
                with self.update_config_lock:
                    current_key = self.settings.get("keybind", 164)
                    mode = self.settings.get("trigger_mode", "hold")
                if mode == "hold":
                    self.active = (win32api.GetAsyncKeyState(current_key) < 0) and not self.paused
                else:
                    current_state = (win32api.GetAsyncKeyState(current_key) < 0)
                    if current_state and not getattr(self, 'last_key_state', False):
                        self.toggled_active = not getattr(self, 'toggled_active', False)
                        log_event(f"Toggle mode switched to: {self.toggled_active}", "INFO", context="TRIGGERBOT", dedup_key="toggle")
                        time.sleep(0.3)
                    self.last_key_state = current_state
                    self.active = getattr(self, 'toggled_active', False) and not self.paused
                time.sleep(0.01)
                now = time.time()
                if now - self.last_log_time > 5:
                    log_event(f"TriggerBot (ScanningManager) Color detections: {self.color_detected_count}", "DEBUG", context=f"MODE={self.settings.get('trigger_mode','hold').upper()} DEVICE={'GPU' if self.use_gpu else 'CPU'}", dedup_key="perf")
                    self.last_log_time = now
            log_event("TriggerBot thread stopped", "INFO", context="TRIGGERBOT")
        except Exception as e:
            log_event(f"Error in TriggerBot thread: {str(e)}", "ERROR", context="TRIGGERBOT")
            traceback.print_exc()
        finally:
            self.manager.unregister_subscriber(self.region_name, self._on_detection_result)

    def stop(self):
        debug_log("Stopping TriggerBot...")
        self.stop_flag = True
        debug_log("TriggerBot stopped")

# Create a PyQt5 compatible QThread version of TriggerBotThread if PyQt5 is available
if PYQT5_AVAILABLE:
    class TriggerBotThread(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        
        def __init__(self, settings, fov, min_shooting_delay_ms, max_shooting_delay_ms, fps, hsv_range, parent=None):
            super().__init__(parent)
            self.settings = settings
            self.fov = fov
            self.min_shooting_delay_ms = settings.get("min_shooting_delay_ms", 50.0)
            self.max_shooting_delay_ms = settings.get("max_shooting_delay_ms", 80.0)
            self.fps = fps
            self.hsv_range = hsv_range
            self.triggerbot = None
            self.shoot_queue = Queue()
            self.shoot_process = None
            self.config_lock = threading.Lock()
            self.is_stopping = False

        def run(self):
            try:
                self.is_stopping = False
                self.shoot_process = Process(target=simulate_shoot, args=(self.shoot_queue, self.settings))
                self.shoot_process.start()
                self.triggerbot = Triggerbot(self.shoot_queue, self.settings, self.fov, self.hsv_range, self.min_shooting_delay_ms, self.max_shooting_delay_ms, self.fps)
                self.triggerbot.run()
            except Exception as e:
                debug_log(f"Error in TriggerBotThread.run: {e}")
                traceback.print_exc()

        def update_config(self, settings):
            with self.config_lock:
                self.settings = settings
                self.fov = settings.get("fov", 5.0)
                self.min_shooting_delay_ms = settings.get("min_shooting_delay_ms", 50.0)
                self.max_shooting_delay_ms = settings.get("max_shooting_delay_ms", 80.0)
                self.hsv_range = settings.get("hsv_range", [[30,125,150],[30,255,255]])
                
                if self.triggerbot:
                    self.triggerbot.update_config(settings)
                    self.log_signal.emit("TriggerBot settings updated")
                
                try:
                    self.shoot_queue.put("UpdateConfig")
                    self.shoot_queue.put(settings)
                    self.log_signal.emit("Shooting process settings updated")
                except Exception as e:
                    self.log_signal.emit(f"Error updating shooting process: {str(e)}")

        def stop(self):
            if self.is_stopping:
                debug_log("Already stopping TriggerBotThread, ignoring duplicate stop request")
                return

            self.is_stopping = True
            try:
                shutdown_event.set()
                
                if self.triggerbot:
                    self.triggerbot.stop_flag = True
                    if hasattr(self.triggerbot, 'camera') and self.triggerbot.camera:
                        try:
                            self.triggerbot.camera.stop()
                        except:
                            pass
                    self.triggerbot = None
                
                if self.shoot_process and self.shoot_process.is_alive():
                    try:
                        self.shoot_process.terminate()
                        self.shoot_process.join(timeout=1.0)
                    except:
                        pass
                    self.shoot_process = None
                    
                shutdown_event.clear()
            except Exception as e:
                debug_log(f"Error stopping TriggerBotThread: {e}")
else:
    # Fallback to threading.Thread if PyQt5 is not available
    class TriggerBotThread(threading.Thread):
        """
        Thread class for running the TriggerBot in a separate thread.
        This is a fallback version that works without PyQt5.
        """
        def __init__(self, settings, fov, min_shooting_delay_ms, max_shooting_delay_ms, fps, hsv_range, parent=None):
            super().__init__()
            self.settings = settings
            self.fov = fov
            self.min_shooting_delay_ms = settings.get("min_shooting_delay_ms", 50.0)
            self.max_shooting_delay_ms = settings.get("max_shooting_delay_ms", 80.0)
            self.fps = fps
            self.hsv_range = hsv_range
            self.triggerbot = None
            self.shoot_queue = Queue()
            self.shoot_process = None
            self.config_lock = threading.Lock()
            self._stop_event = threading.Event()
            self.is_stopping = False
            
            # Create a dummy log_signal
            self.log_signal = None
            
        def log(self, message):
            debug_log(message)

        def run(self):
            try:
                self._stop_event.clear()
                self.is_stopping = False
                self.shoot_process = Process(target=simulate_shoot, args=(self.shoot_queue, self.settings))
                self.shoot_process.start()
                self.triggerbot = Triggerbot(self.shoot_queue, self.settings, self.fov, self.hsv_range, self.min_shooting_delay_ms, self.max_shooting_delay_ms, self.fps)
                self.triggerbot.run()
            except Exception as e:
                debug_log(f"Error in TriggerBotThread.run: {e}")
                traceback.print_exc()

        def update_config(self, settings):
            with self.config_lock:
                self.settings = settings
                self.fov = settings.get("fov", 5.0)
                self.min_shooting_delay_ms = settings.get("min_shooting_delay_ms", 50.0)
                self.max_shooting_delay_ms = settings.get("max_shooting_delay_ms", 80.0)
                self.hsv_range = settings.get("hsv_range", [[30,125,150],[30,255,255]])
                
                if self.triggerbot:
                    self.triggerbot.update_config(settings)
                    self.log("TriggerBot settings updated")
                
                try:
                    self.shoot_queue.put("UpdateConfig")
                    self.shoot_queue.put(settings)
                    self.log("Shooting process settings updated")
                except Exception as e:
                    self.log(f"Error updating shooting process: {str(e)}")

        def stop(self):
            if self.is_stopping:
                debug_log("Already stopping TriggerBotThread, ignoring duplicate stop request")
                return

            self.is_stopping = True
            if self.triggerbot:
                self.triggerbot.stop()
                self.triggerbot = None
            
            if self.shoot_process and self.shoot_process.is_alive():
                shutdown_event.set()
                try:
                    self.shoot_process.join(1.0)
                    if self.shoot_process.is_alive():
                        self.shoot_process.terminate()
                except:
                    pass
                self.shoot_process = None
            
            self._stop_event.set()
            shutdown_event.clear()
            debug_log("TriggerBotThread stopped")

        def isRunning(self):
            return not self._stop_event.is_set() 

# --- Compatibility re-exports for menu.py and legacy imports ---
from shared_scanner import SharedScanner

def detect_color(*args, **kwargs):
    """Legacy stub. Use SharedScanner().scan instead."""
    raise NotImplementedError("detect_color is deprecated. Use SharedScanner().scan instead.")

def detect_color_pytorch(*args, **kwargs):
    """Legacy stub. Use SharedScanner().detect_color_gpu instead."""
    raise NotImplementedError("detect_color_pytorch is deprecated. Use SharedScanner().detect_color_gpu instead.") 