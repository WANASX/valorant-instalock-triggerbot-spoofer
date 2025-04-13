import sys, os, time, random, ctypes, json, threading
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2, numpy as np, pyautogui
import win32api, win32con
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QTabWidget, QSplitter, QStatusBar, QToolTip, QStyleFactory
import bettercam
from tempfile import gettempdir
from PIL import Image
import traceback
import queue

# Enable debug mode
DEBUG_MODE = True

def debug_log(message):
    """Log debug messages to console if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - {message}")

# Check if OpenCV was compiled with CUDA
def check_opencv_cuda_support():
    try:
        # Try to create a CUDA-based function
        test_array = np.zeros((10, 10), dtype=np.uint8)
        test_gpu_mat = cv2.cuda_GpuMat()
        test_gpu_mat.upload(test_array)
        test_gpu_mat.release()
        return True
    except cv2.error as e:
        if "no cuda support" in str(e).lower():
            debug_log("OpenCV was not compiled with CUDA support")
            return False
        else:
            debug_log(f"Other OpenCV error: {e}")
            return False
    except Exception as e:
        debug_log(f"Error testing CUDA support: {e}")
        return False

# Improved CUDA detection
CUDA_AVAILABLE = False
OPENCV_HAS_CUDA = False
try:
    # First check if OpenCV was compiled with CUDA
    OPENCV_HAS_CUDA = hasattr(cv2, 'cuda')
    debug_log(f"OpenCV CUDA module exists: {OPENCV_HAS_CUDA}")
    
    if OPENCV_HAS_CUDA:
        # Force enable CUDA for testing
        debug_log("CUDA module exists - forcing CUDA_AVAILABLE to True for testing")
        CUDA_AVAILABLE = True
        
        # Check if CUDA devices are available - just for logging
        try:
            cv2_cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            debug_log(f"OpenCV CUDA devices detected: {cv2_cuda_count}")
            if cv2_cuda_count == 0:
                debug_log("WARNING: No CUDA devices detected, but continuing with CUDA enabled")
        except Exception as e:
            debug_log(f"Error checking CUDA devices: {e}")
    else:
        debug_log("WARNING: OpenCV was not compiled with CUDA support")
        debug_log("To use CUDA acceleration, install OpenCV with CUDA support:")
        debug_log("pip uninstall opencv-python opencv-python-headless")
        debug_log("pip install opencv-python-cuda")
        
        # Try detecting CUDA from common installation paths
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            debug_log(f"CUDA found in environment at: {cuda_path}")
            debug_log("CUDA is installed but OpenCV can't use it")
        else:
            possible_paths = [
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
                r'C:\Program Files\NVIDIA\CUDA',
                r'C:\CUDA'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    debug_log(f"CUDA installation found at: {path}")
                    debug_log("CUDA is installed but OpenCV can't use it")
                    break
        
except Exception as e:
    debug_log(f"Error during CUDA detection: {e}")
    traceback.print_exc()

# Override CUDA availability - force it on for testing
if hasattr(cv2, 'cuda'):
    debug_log("OVERRIDING CUDA DETECTION: Forcing CUDA to be available for testing")
    CUDA_AVAILABLE = True
    OPENCV_HAS_CUDA = True

# Import PyTorch for better GPU support
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        debug_log(f"PyTorch CUDA available: {TORCH_AVAILABLE}, Device: {torch.cuda.get_device_name(0)}")
    else:
        debug_log("PyTorch CUDA not available")
except ImportError:
    TORCH_AVAILABLE = False
    debug_log("PyTorch not installed, falling back to OpenCV only")

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
                return json.load(f)
        except Exception:
            pass
    # Default configuration with new press_duration and block movements option
    return {
        "fov": 5.0,
        "keybind": 164,
        "shooting_rate": 65.0,
        "min_shooting_rate": 50.0,  # Minimum shooting delay in ms
        "max_shooting_rate": 80.0,  # Maximum shooting delay in ms
        "fps": 200.0,
        "hsv_range": [[30, 125, 150], [30, 255, 255]],
        "trigger_mode": "hold",  # or "toggle"
        "press_duration_min": 0.01,  # Minimum press duration in seconds
        "press_duration_max": 0.05,  # Maximum press duration in seconds
        "enable_press_duration": True,  # Toggle for using random press duration
        "block_movements": False,   # If True, block (W, A, S, D) while shooting
        "use_gpu": CUDA_AVAILABLE,  # Use GPU if available
        "auto_fallback_to_cpu": True,  # Auto fallback to CPU if GPU fails
        "smart_acceleration": True,  # Automatically select best acceleration method
        "test_mode": False,  # Test mode for comparing GPU/CPU
        "theme": "custom"  # Default theme set to custom green
    }

# ------------------- TriggerBot Logic -------------------

def simulate_shoot(q, config):
    keybd_event = ctypes.windll.user32.keybd_event
    while not shutdown_event.is_set():
        try:
            # Use a small timeout to avoid blocking the loop but handle Empty exception silently
            try:
                signal_value = q.get(timeout=0.1)
            except queue.Empty:
                # This is normal - just wait for next signal
                continue
            
            # Handle config update message
            if signal_value == "UpdateConfig":
                try:
                    # Get the updated config from the queue
                    updated_config = q.get(timeout=0.1)
                    # Update the local config reference with the new values
                    config.update(updated_config)
                    debug_log("Shooting process config updated: " + 
                             f"block_movements={config.get('block_movements', False)}, " +
                             f"enable_press_duration={config.get('enable_press_duration', True)}, " +
                             f"press_duration_min={config.get('press_duration_min', 0.01)}, " +
                             f"press_duration_max={config.get('press_duration_max', 0.03)}")
                    continue
                except Exception as e:
                    debug_log(f"Error updating config in shoot process: {type(e).__name__}: {str(e)}")
                    continue
            
            if signal_value == "Shoot":
                if config.get("enable_press_duration", True):
                    press_duration_min = config.get("press_duration_min", 0.01)
                    press_duration_max = config.get("press_duration_max", 0.05)
                    press_duration = random.uniform(press_duration_min, press_duration_max)
                else:
                    press_duration = 0  # No delay if random press duration is disabled

                # If blocking movements is enabled, release W, A, S, D keys
                if config.get("block_movements", False):
                    for key in [0x57, 0x41, 0x53, 0x44]:
                        if win32api.GetAsyncKeyState(key) < 0:
                            keybd_event(key, 0, 2, 0)  # simulate key up
                keybd_event(0x01, 0, 0, 0)
                time.sleep(press_duration)
                keybd_event(0x01, 0, 2, 0)
        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)}"
            debug_log(f"Error in shoot process: {error_info}")
            # Print traceback for more detailed debugging
            traceback.print_exc()
            continue

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
            "opencv_gpu_time": float('inf'),
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
        
        # OpenCV GPU test (may not work if OpenCV isn't compiled with CUDA)
        if OPENCV_HAS_CUDA and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            try:
                debug_log("Running OpenCV GPU benchmark...")
                opencv_gpu_start = time.time()
                
                for _ in range(test_iterations):
                    # Upload to GPU
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame_large)
                    
                    # Convert to HSV on GPU
                    gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2HSV)
                    
                    # Try multiple operations to stress the GPU
                    # Apply color thresholding
                    for h in range(0, 180, 5):  # More color checks
                        lower = cv2.cuda_GpuMat()
                        upper = cv2.cuda_GpuMat()
                        lower.upload(np.array([h, 50, 50], dtype=np.uint8))
                        upper.upload(np.array([h+5, 255, 255], dtype=np.uint8))
                        gpu_mask = cv2.cuda.inRange(gpu_hsv, lower, upper)
                        
                        # Add more GPU operations
                        if h % 20 == 0:  # For some of the masks, do more processing
                            # Apply morphological operations
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                            gpu_mask_dilated = cv2.cuda.dilate(gpu_mask, kernel)
                            gpu_mask_eroded = cv2.cuda.erode(gpu_mask_dilated, kernel)
                        
                        # Download result (needed for practical use)
                        mask_result = gpu_mask.download()
                        has_color = np.any(mask_result)
                        
                        # Clean up
                        lower.release()
                        upper.release()
                        gpu_mask.release()
                        if h % 20 == 0:
                            gpu_mask_dilated.release()
                            gpu_mask_eroded.release()
                    
                    # Clean up
                    gpu_frame.release()
                    gpu_hsv.release()
                
                opencv_gpu_time = (time.time() - opencv_gpu_start) * 1000 / test_iterations
                debug_log(f"OpenCV GPU average time: {opencv_gpu_time:.2f}ms")
                results["opencv_gpu_time"] = max(opencv_gpu_time, 0.0001)  # Prevent zero
                
            except Exception as e:
                debug_log(f"OpenCV GPU benchmark failed: {str(e)}")
        else:
            debug_log("OpenCV GPU benchmark skipped: CUDA not available in OpenCV")
        
        # CPU test (using OpenCV)
        debug_log("Running OpenCV CPU benchmark...")
        cpu_start = time.time()
        for _ in range(test_iterations):
            hsv = cv2.cvtColor(frame_large, cv2.COLOR_RGB2HSV)
            
            # Do multiple operations to match GPU workload
            for h in range(0, 180, 5):
                # Create a test mask for each hue range
                lower = np.array([h, 50, 50], dtype=np.uint8)
                upper = np.array([h+5, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                has_color = np.any(mask)
                
                # Add more operations to match GPU workload
                if h % 20 == 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    mask_dilated = cv2.dilate(mask, kernel)
                    mask_eroded = cv2.erode(mask_dilated, kernel)
        
        cpu_time = (time.time() - cpu_start) * 1000 / test_iterations
        debug_log(f"CPU average time: {cpu_time:.2f}ms")
        results["cpu_time"] = max(cpu_time, 0.0001)  # Prevent zero
        
        # Determine best GPU method
        best_gpu_time = min(results["opencv_gpu_time"], results["pytorch_gpu_time"])
        results["best_gpu_time"] = best_gpu_time
        
        if best_gpu_time == results["pytorch_gpu_time"] and results["pytorch_gpu_time"] < float('inf'):
            results["best_method"] = "PyTorch GPU"
        elif best_gpu_time == results["opencv_gpu_time"] and results["opencv_gpu_time"] < float('inf'):
            results["best_method"] = "OpenCV GPU"
        else:
            results["best_method"] = "CPU"
        
        # Calculate speedup ratio with protection against div by 0
        if best_gpu_time < float('inf'):
            if best_gpu_time > 0:
                results["ratio"] = results["cpu_time"] / best_gpu_time
            else:
                results["ratio"] = 100  # Arbitrarily large if GPU time is zero/near-zero
                
            if results["ratio"] > 1:
                debug_log(f"{results['best_method']} is {results['ratio']:.2f}x faster than CPU")
            else:
                slowdown = 1.0 / max(results["ratio"], 0.0001)  # Avoid div by 0
                debug_log(f"WARNING: {results['best_method']} is {slowdown:.2f}x SLOWER than CPU!")
        else:
            results["ratio"] = 0
            debug_log("No GPU methods available for comparison")
        
        # Log complete results
        debug_log(f"Benchmark complete: CPU={results['cpu_time']:.2f}ms, PyTorch={results['pytorch_gpu_time']:.2f}ms, OpenCV GPU={results['opencv_gpu_time']:.2f}ms")
        debug_log(f"Best method: {results['best_method']}")
        
        return results
        
    except Exception as e:
        debug_log(f"Error during benchmark: {str(e)}")
        traceback.print_exc()
        return {"cpu_time": 0.0001, "opencv_gpu_time": float('inf'), "pytorch_gpu_time": float('inf'), 
                "best_gpu_time": float('inf'), "ratio": 0, "best_method": "CPU (benchmark failed)"}

# Add a new function for PyTorch-based color detection right before the detect_color function
def detect_color_pytorch(frame, cmin, cmax):
    """
    Detect if the specified color range exists in the frame using PyTorch with CUDA.
    This is a dedicated PyTorch implementation optimized for GPU performance.
    
    Args:
        frame: The image frame to check (numpy array)
        cmin: Minimum HSV values [h_min, s_min, v_min]
        cmax: Maximum HSV values [h_max, s_max, v_max]
    
    Returns:
        Boolean indicating if the color was detected, None if error occurred
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
        
    try:
        h_min, s_min, v_min = cmin
        h_max, s_max, v_max = cmax
        
        # Use non-blocking transfer for better performance
        with torch.amp.autocast('cuda', enabled=True):  # Using updated API syntax
            # Convert to PyTorch tensor and move to GPU - use non-blocking for better performance
            frame_tensor = torch.from_numpy(frame).cuda(non_blocking=True).float() / 255.0
            
            # Get the dimensions of the tensor
            if len(frame_tensor.shape) == 3:  # [H, W, C]
                # Reorder channels from [H, W, C] to [C, H, W]
                frame_tensor = frame_tensor.permute(2, 0, 1)
            
            # Extract RGB channels
            if frame_tensor.shape[0] == 3:  # Ensure we have 3 channels
                r, g, b = frame_tensor[0], frame_tensor[1], frame_tensor[2]
                
                # Calculate HSV values using PyTorch operations
                max_val, _ = torch.max(frame_tensor, dim=0)
                min_val, _ = torch.min(frame_tensor, dim=0)
                diff = max_val - min_val
                
                # Value is max_val
                v = max_val
                
                # Saturation is diff / max_val (or 0 if max_val is 0)
                s = torch.where(max_val != 0, diff / max_val, torch.zeros_like(max_val))
                
                # Hue calculation
                h = torch.zeros_like(max_val)
                
                # Create masks for different max channels
                r_max_mask = (r == max_val) & (diff != 0)
                g_max_mask = (g == max_val) & (diff != 0)
                b_max_mask = (b == max_val) & (diff != 0)
                
                # Calculate hue based on which channel is max
                h[r_max_mask] = (60 * ((g[r_max_mask] - b[r_max_mask]) / diff[r_max_mask]) + 360) % 360
                h[g_max_mask] = (60 * ((b[g_max_mask] - r[g_max_mask]) / diff[g_max_mask]) + 120)
                h[b_max_mask] = (60 * ((r[b_max_mask] - g[b_max_mask]) / diff[b_max_mask]) + 240)
                
                # Scale to OpenCV ranges [0-179, 0-255, 0-255]
                h = h / 2  # OpenCV uses [0, 179] for Hue
                s = s * 255
                v = v * 255
                
                # Create a binary mask where values are in the target color range
                mask = (h >= h_min) & (h <= h_max) & (s >= s_min) & (s <= s_max) & (v >= v_min) & (v <= v_max)
                
                # Check if any values in mask are True - use .any() for best performance
                # Only synchronize at this final step
                result = torch.any(mask).item()
            
                # Force CUDA to complete all operations
                torch.cuda.synchronize()
        
        # Only do cleanup if we're experiencing memory pressure (no need for every frame)
        if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
            del frame_tensor, r, g, b, max_val, min_val, diff, v, s, h, r_max_mask, g_max_mask, b_max_mask, mask
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        debug_log(f"PyTorch color detection error: {str(e)}")
        # Clean up and release CUDA memory on error
        try:
            torch.cuda.empty_cache()
        except:
            pass
        return None

def detect_color(frame, cmin, cmax, use_gpu=False, triggerbot=None, force_cpu_mode=False):
    """
    Detect if the specified color range exists in the frame.
    
    Args:
        frame: The image frame to check
        cmin: Minimum HSV values [h_min, s_min, v_min]
        cmax: Maximum HSV values [h_max, s_max, v_max]
        use_gpu: Whether to use GPU acceleration
        triggerbot: Reference to the Triggerbot instance for auto-fallback
        force_cpu_mode: If True, always use CPU mode (for testing)
    
    Returns:
        Boolean indicating if the color was detected
    """
    if frame is None:
        debug_log("detect_color: Frame is None")
        return False
    
    start_time = time.time()
    result = False
    mode_used = "CPU"
    
    # Try PyTorch GPU method if GPU is enabled and not forced to CPU
    if use_gpu and not force_cpu_mode:
        # Attempt PyTorch detection (optimized GPU method)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            debug_log("Attempting PyTorch GPU detection")
            torch_result = detect_color_pytorch(frame, cmin, cmax)
            
            if torch_result is not None:
                result = torch_result
                mode_used = "GPU-PyTorch"
                process_time = (time.time() - start_time) * 1000
                debug_log(f"PyTorch GPU detection result: {result}, took {process_time:.2f}ms")
                
                # Reset GPU failure count on success
                if triggerbot is not None:
                    triggerbot.gpu_failure_count = 0
                
                return result
            else:
                debug_log("PyTorch detection failed, trying OpenCV")
                
                # Track failures for auto-fallback
                if triggerbot is not None:
                    triggerbot.gpu_failure_count += 1
    
    # If we got here, either GPU mode is disabled, force_cpu_mode is True,
    # or the PyTorch method failed, so use OpenCV CPU method
    try:
        debug_log("Using OpenCV CPU color detection")
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(cmin, dtype=np.uint8), np.array(cmax, dtype=np.uint8))
        result = np.any(mask)
        mode_used = "CPU-OpenCV"
        
        # Auto-fallback to CPU if too many GPU failures
        if triggerbot is not None and triggerbot.auto_fallback and triggerbot.gpu_failure_count >= triggerbot.max_gpu_failures:
            debug_log(f"Too many GPU failures ({triggerbot.gpu_failure_count}). Switching to CPU mode permanently.")
            triggerbot.use_gpu = False
    except Exception as e:
        debug_log(f"Error in OpenCV CPU color detection: {e}")
        return False
    
    process_time = (time.time() - start_time) * 1000
    debug_log(f"{mode_used} color detection result: {result}, took {process_time:.2f}ms")
    return result

class Triggerbot:
    def __init__(self, q, settings, fov, hsv_range, shooting_rate, fps):
        self.queue = q
        self.settings = settings  # shared config dictionary
        self.shooting_rate = shooting_rate / 1000.0
        
        # Initialize min/max shooting rates
        self.min_shooting_rate = settings.get("min_shooting_rate", 50.0) / 1000.0
        self.max_shooting_rate = settings.get("max_shooting_rate", 80.0) / 1000.0
        
        self.fps = int(fps)
        self.fov = int(fov)
        self.hsv_range = hsv_range
        self.frames_processed = 0
        self.last_shot_time = 0
        self.check_region = None
        self.camera = None
        self.stop_flag = False
        self.paused = False
        self.region_changed = True  # Force region update on startup
        self.use_gpu = settings.get("use_gpu", False) and CUDA_AVAILABLE
        self.auto_fallback = settings.get("auto_fallback_to_cpu", True)
        self.test_mode = settings.get("test_mode", False)
        self.max_gpu_failures = 3
        self.gpu_failure_count = 0
        self.config_updated = False
        self.force_cpu_mode = False  # Add the missing attribute
        self.camera_lock = threading.Lock()  # Add lock for thread-safe camera access
        
        # Get screen dimensions for center calculation
        user32 = ctypes.windll.user32
        self.WIDTH, self.HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        
        # Parse HSV range
        cmin = self.hsv_range[0]
        cmax = self.hsv_range[1]
        self.cmin = cmin
        self.cmax = cmax
        
        self.update_config_lock = threading.Lock()
        
        # Initialize check region and camera
        self.update_check_region()
        
        # Start with a valid camera
        try:
            self.camera = bettercam.create(output_idx=0, region=self.check_region)
            self.camera.start(target_fps=self.fps)
            debug_log(f"Camera created with initial region: {self.check_region}")
        except Exception as e:
            debug_log(f"Error creating initial camera: {e}")
            traceback.print_exc()
        
        debug_log(f"Triggerbot initialized with GPU={self.use_gpu}, Auto-fallback={self.auto_fallback}")

    def update_check_region(self):
        """Update the region to check based on the current FOV"""
        try:
            # Calculate region based on FOV and screen center
            center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
            
            # Use the full FOV value, not half
            fov_size = max(4, self.fov)  # Ensure at least 4 pixels for detection
            
            left = center_x - fov_size
            top = center_y - fov_size
            right = center_x + fov_size
            bottom = center_y + fov_size
            
            # Clamp to screen bounds
            left = max(0, left)
            top = max(0, top)
            right = min(self.WIDTH, right)
            bottom = min(self.HEIGHT, bottom)
            
            # Save the new region
            self.check_region = (left, top, right, bottom)
            debug_log(f"Updated check region to {self.check_region}")
            
            # Don't automatically update the camera here
            # Just set the flag so the main loop will recreate it
            self.region_changed = True
            
        except Exception as e:
            debug_log(f"Error updating check region: {e}")
            traceback.print_exc()

    def update_hsv_range(self, hsv_range):
        """Update HSV range used for color detection"""
        self.hsv_range = hsv_range
        self.cmin = hsv_range[0]
        self.cmax = hsv_range[1]
        debug_log(f"Updated HSV range to {hsv_range}")
        
        # Immediately apply the change
        self.config_updated = True
    
    def update_config(self, settings):
        with self.update_config_lock:
            # Store old FOV to check if it changed
            old_fov = self.fov
            
            # Update settings
            self.settings = settings
            self.fov = int(settings.get("fov", 5.0))
            
            # Update shooting rate with min and max values
            min_shooting_rate = settings.get("min_shooting_rate", 50.0) / 1000.0
            max_shooting_rate = settings.get("max_shooting_rate", 80.0) / 1000.0
            self.min_shooting_rate = min_shooting_rate
            self.max_shooting_rate = max_shooting_rate
            # Keep for backward compatibility
            self.shooting_rate = settings.get("shooting_rate", 65.0) / 1000.0
            
            self.use_gpu = settings.get("use_gpu", False) and CUDA_AVAILABLE
            self.auto_fallback = settings.get("auto_fallback_to_cpu", True)
            
            # Log important setting changes for debugging
            block_movements = settings.get("block_movements", False)
            enable_press_duration = settings.get("enable_press_duration", True)
            press_min = settings.get("press_duration_min", 0.01)
            press_max = settings.get("press_duration_max", 0.05)
            debug_log(f"Triggerbot settings updated: block_movements={block_movements}, enable_press_duration={enable_press_duration}, press_min={press_min}, press_max={press_max}")
            debug_log(f"Shooting rate range: {min_shooting_rate*1000:.0f}ms - {max_shooting_rate*1000:.0f}ms")
            
            # Special settings for test mode
            if hasattr(self, 'test_mode'):
                self.test_mode = settings.get("test_mode", False)
                if self.test_mode:
                    debug_log("TEST MODE ENABLED: Will alternate between GPU and CPU for performance testing")
            
            debug_log(f"Config updated: GPU={self.use_gpu}, Auto-fallback={self.auto_fallback}")
            
            # Only update check region if FOV changed
            if old_fov != self.fov:
                debug_log(f"FOV changed from {old_fov} to {self.fov}, updating region immediately")
                self.update_check_region()
                self.region_changed = False  # Reset since we've already updated
            
            self.update_hsv_range(settings.get("hsv_range", [[30,125,150],[30,255,255]]))
            self.config_updated = True

    def run(self):
        """Main triggerbot thread that runs continuously"""
        try:
            debug_log("TriggerBot thread started")
            frames_processed = 0
            color_detected_count = 0
            fps_counter = 0
            last_fps_check = time.time()
            
            # For test mode
            cpu_frames = 0
            gpu_frames = 0
            cpu_time_total = 0
            gpu_time_total = 0
            
            # For auto-benchmarking
            last_benchmark_time = time.time()
            benchmark_complete = False
            best_method = "CPU"
            
            # Initialize with default settings
            use_gpu = self.use_gpu
            force_cpu = self.force_cpu_mode
            
            # Initialize camera
            if not hasattr(self, 'camera') or self.camera is None:
                try:
                    self.camera = bettercam.create(output_idx=0, region=self.check_region)
                    self.camera.start(target_fps=self.fps)
                    debug_log(f"Initial camera created with region: {self.check_region}")
                except Exception as e:
                    debug_log(f"Error creating initial camera: {e}")
            
            while not self.stop_flag and not shutdown_event.is_set():
                # Safely get current settings
                with self.update_config_lock:
                    current_key = self.settings.get("keybind", 164)
                    mode = self.settings.get("trigger_mode", "hold")
                    self.config_updated = False
                
                # Determine if active based on trigger mode
                if mode == "hold":
                    active = (win32api.GetAsyncKeyState(current_key) < 0) and not self.paused
                else:  # Toggle mode
                    current_state = (win32api.GetAsyncKeyState(current_key) < 0)
                    if current_state and not getattr(self, 'last_key_state', False):
                        self.toggled_active = not getattr(self, 'toggled_active', False)
                        debug_log(f"Toggle mode switched to: {self.toggled_active}")
                        time.sleep(0.3)  # debounce delay
                    self.last_key_state = current_state
                    active = getattr(self, 'toggled_active', False) and not self.paused
                
                # Update region if changed - correctly recreate camera
                if self.region_changed:
                    debug_log(f"Region change detected - recreating camera with new region: {self.check_region}")
                    try:
                        # Stop old camera if it exists
                        if self.camera:
                            try:
                                self.camera.stop()
                            except:
                                pass
                        
                        # Delete reference to old camera before creating new one
                        # This is important for bettercam which checks if an instance exists
                        self.camera = None
                        time.sleep(0.2)  # Small delay to ensure resources are released
                        
                        # Create new camera with updated region
                        self.camera = bettercam.create(output_idx=0, region=self.check_region)
                        self.camera.start(target_fps=self.fps)
                        debug_log(f"Camera recreated with region: {self.check_region}")
                    except Exception as e:
                        debug_log(f"Error recreating camera: {e}")
                    
                    self.region_changed = False

                if active:
                    # Get frame safely
                    frame = None
                    if self.camera:
                        try:
                            frame = self.camera.get_latest_frame()
                        except Exception as e:
                            debug_log(f"Error getting frame: {e}")
                            # Clear camera reference so we'll recreate it next time
                            try:
                                if self.camera:
                                    self.camera.stop()
                                self.camera = None
                            except:
                                pass
                    
                    if frame is not None:
                        self.frames_processed += 1
                        frames_processed += 1
                        
                        # Run benchmark periodically (every 10 minutes) or if user changes GPU settings
                        current_time = time.time()
                        if not benchmark_complete or self.config_updated or current_time - last_benchmark_time > 600:
                            # Use fast benchmark for runtime checks to avoid lag
                            debug_log("Running quick performance check...")
                            benchmark_results = fast_benchmark(frame)
                            best_method = benchmark_results["best_method"]
                            
                            # Auto-select best mode if auto fallback is enabled
                            if self.auto_fallback:
                                if best_method == "GPU" and benchmark_results["ratio"] > 1.2:
                                    if not use_gpu:
                                        debug_log(f"Auto-switching to GPU mode (GPU is {benchmark_results['ratio']:.2f}x faster)")
                                        self.use_gpu = True
                                        use_gpu = True
                                elif benchmark_results["ratio"] < 1.0:
                                    if use_gpu:
                                        debug_log(f"Auto-switching to CPU mode (GPU is slower, ratio: {benchmark_results['ratio']:.2f}x)")
                                        self.use_gpu = False
                                        use_gpu = False
                            
                            benchmark_complete = True
                            last_benchmark_time = current_time
                            self.config_updated = False
                        
                        # Color detection
                        start_time = time.time()
                        color_detected = detect_color(frame, self.cmin, self.cmax, use_gpu, self, force_cpu)
                        elapsed = (time.time() - start_time) * 1000
                        
                        # Track stats for test mode
                        if self.test_mode:
                            if force_cpu:
                                cpu_frames += 1
                                cpu_time_total += elapsed
                            else:
                                gpu_frames += 1
                                gpu_time_total += elapsed
                        
                        if color_detected:
                            color_detected_count += 1
                            debug_log(f"Target color detected (frame #{frames_processed})!")
                            current_time = time.time()
                            # Use random delay between min and max
                            random_delay = random.uniform(self.min_shooting_rate, self.max_shooting_rate)
                            if current_time - self.last_shot_time >= random_delay:
                                debug_log(f"Shoot signal sent (delay: {random_delay*1000:.0f}ms)")
                                self.queue.put("Shoot")
                                self.last_shot_time = current_time
                        else:
                            debug_log("Failed to get frame from camera")
                            
                        time.sleep(1 / self.fps)
                else:
                    time.sleep(0.01)
                    
                # Calculate and print FPS every 5 seconds
                fps_counter += 1
                if time.time() - last_fps_check >= 5:
                    elapsed = time.time() - last_fps_check
                    fps = fps_counter / elapsed
                    debug_log(f"TriggerBot FPS: {fps:.1f}, Frames processed: {frames_processed}, Color detections: {color_detected_count}")
                    debug_log(f"Current acceleration mode: {best_method if use_gpu else 'CPU'}")
                    
                    # Show test mode stats if active
                    if self.test_mode and (gpu_frames > 0 or cpu_frames > 0):
                        gpu_avg = gpu_time_total / gpu_frames if gpu_frames > 0 else 0
                        cpu_avg = cpu_time_total / cpu_frames if cpu_frames > 0 else 0
                        debug_log(f"TEST MODE STATS - GPU: {gpu_avg:.2f}ms ({gpu_frames} frames), CPU: {cpu_avg:.2f}ms ({cpu_frames} frames)")
                    
                    # Reset counters
                    fps_counter = 0
                    last_fps_check = time.time()
                    
            debug_log("TriggerBot thread stopped")
        except Exception as e:
            debug_log(f"Error in TriggerBot thread: {str(e)}")
            traceback.print_exc()

    def stop(self):
        debug_log("Stopping TriggerBot...")
        self.stop_flag = True
        with self.camera_lock:
            if self.camera:
                try:
                    self.camera.stop()
                    self.camera = None
                except:
                    pass
        debug_log("TriggerBot stopped")

    def update_config(self, settings):
        with self.update_config_lock:
            # Store old FOV to check if it changed
            old_fov = self.fov
            
            # Update settings
            self.settings = settings
            self.fov = int(settings.get("fov", 5.0))
            
            # Update shooting rate with min and max values
            min_shooting_rate = settings.get("min_shooting_rate", 50.0) / 1000.0
            max_shooting_rate = settings.get("max_shooting_rate", 80.0) / 1000.0
            self.min_shooting_rate = min_shooting_rate
            self.max_shooting_rate = max_shooting_rate
            # Keep for backward compatibility
            self.shooting_rate = settings.get("shooting_rate", 65.0) / 1000.0
            
            self.use_gpu = settings.get("use_gpu", False) and CUDA_AVAILABLE
            self.auto_fallback = settings.get("auto_fallback_to_cpu", True)
            
            # Log important setting changes for debugging
            block_movements = settings.get("block_movements", False)
            enable_press_duration = settings.get("enable_press_duration", True)
            press_min = settings.get("press_duration_min", 0.01)
            press_max = settings.get("press_duration_max", 0.05)
            debug_log(f"Triggerbot settings updated: block_movements={block_movements}, enable_press_duration={enable_press_duration}, press_min={press_min}, press_max={press_max}")
            debug_log(f"Shooting rate range: {min_shooting_rate*1000:.0f}ms - {max_shooting_rate*1000:.0f}ms")
            
            # Special settings for test mode
            if hasattr(self, 'test_mode'):
                self.test_mode = settings.get("test_mode", False)
                if self.test_mode:
                    debug_log("TEST MODE ENABLED: Will alternate between GPU and CPU for performance testing")
            
            debug_log(f"Config updated: GPU={self.use_gpu}, Auto-fallback={self.auto_fallback}")
            
            # Only update check region if FOV changed
            if old_fov != self.fov:
                debug_log(f"FOV changed from {old_fov} to {self.fov}, updating region immediately")
                self.update_check_region()
                self.region_changed = False  # Reset since we've already updated
            
            self.update_hsv_range(settings.get("hsv_range", [[30,125,150],[30,255,255]]))
            self.config_updated = True

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

class TriggerBotThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    def __init__(self, settings, fov, shooting_rate, fps, hsv_range, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.fov = fov
        self.shooting_rate = shooting_rate
        self.min_shooting_rate = settings.get("min_shooting_rate", 50.0)
        self.max_shooting_rate = settings.get("max_shooting_rate", 80.0)
        self.fps = fps
        self.hsv_range = hsv_range
        self.triggerbot = None
        self.shoot_queue = Queue()
        self.shoot_process = None
        self.config_lock = threading.Lock()

    def run(self):
        # Start the shooting process
        self.shoot_process = Process(target=simulate_shoot, args=(self.shoot_queue, self.settings))
        self.shoot_process.start()
        
        # Create and run the triggerbot
        self.triggerbot = Triggerbot(self.shoot_queue, self.settings, self.fov, self.hsv_range, self.shooting_rate, self.fps)
        self.triggerbot.run()

    def update_config(self, settings):
        with self.config_lock:
            self.settings = settings
            self.fov = settings.get("fov", 5.0)
            self.shooting_rate = settings.get("shooting_rate", 65.0)
            self.min_shooting_rate = settings.get("min_shooting_rate", 50.0)
            self.max_shooting_rate = settings.get("max_shooting_rate", 80.0)
            self.hsv_range = settings.get("hsv_range", [[30,125,150],[30,255,255]])
            
            # Update triggerbot if running
            if self.triggerbot:
                self.triggerbot.update_config(settings)
                self.log_signal.emit("TriggerBot settings updated")
            
            # Send settings update to the shoot process
            try:
                # Pass updated settings to the shooting process
                self.shoot_queue.put("UpdateConfig")  # Signal that config will be updated
                self.shoot_queue.put(settings)  # Send the actual config
                self.log_signal.emit("Shooting process settings updated")
            except Exception as e:
                self.log_signal.emit(f"Error updating shooting process: {str(e)}")

    def stop(self):
        try:
            # Signal the global shutdown
            shutdown_event.set()
            
            # Stop the triggerbot
            if self.triggerbot:
                self.triggerbot.stop_flag = True
                if hasattr(self.triggerbot, 'camera') and self.triggerbot.camera:
                    try:
                        self.triggerbot.camera.stop()
                    except:
                        pass
            
            # Stop the shoot process
            if self.shoot_process and self.shoot_process.is_alive():
                self.shoot_process.terminate()
                self.shoot_process.join(timeout=1.0)
                
            # Terminate this thread
            self.terminate()
        except Exception as e:
            debug_log(f"Error stopping TriggerBotThread: {e}")

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
        self.config = load_config()
        
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
        
        self.init_ui()
        
        # Status bar with GPU info
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Always show CUDA as available since the user has confirmed CUDA 12.4
        self.statusBar.showMessage("CUDA 12.4 GPU Acceleration Available")
            
        # Set theme based on config (now defaulting to custom green)
        self.apply_theme(self.config.get("theme", "custom"))

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
        header_label = QtWidgets.QLabel("GamerFun Valo Menu V3")
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
        
        # Create tabs
        self.instalock_tab = QtWidgets.QWidget()
        self.triggerbot_tab = QtWidgets.QWidget()
        self.settings_tab = QtWidgets.QWidget()
        self.logs_tab = QtWidgets.QWidget()
        
        # Add tabs to widget
        self.tabs.addTab(self.instalock_tab, "Agent Instalock")
        self.tabs.addTab(self.triggerbot_tab, "TriggerBot")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.logs_tab, "Logs")
        
        main_layout.addWidget(self.tabs)
        
        # Setup each tab
        self.setup_instalock_tab()
        self.setup_triggerbot_tab()
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
        self.config["enable_press_duration"] = enabled
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
        new_fov = self.fov_spin.value()
        
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
        
        self.config["min_shooting_rate"] = min_rate
        self.config["max_shooting_rate"] = max_rate
        self.config["shooting_rate"] = (min_rate + max_rate) / 2  # Average for backward compatibility
        
        self.config["press_duration_min"] = self.press_duration_min_spin.value()
        self.config["press_duration_max"] = self.press_duration_max_spin.value()
        self.config["enable_press_duration"] = self.press_duration_toggle.isChecked()
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
                
                # We need to explicitly stop and recreate the camera because 
                # of how the triggerbot's run loop may not detect the region_changed flag quickly
                try:
                    debug_log("FOV changed - forcing immediate camera recreation")
                    
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
                    
                    # Update the check region - ensure all values are integers
                    triggerbot.check_region = (int(left), int(top), int(right), int(bottom))
                    debug_log(f"New check region: {triggerbot.check_region}")
                    
                    # Stop old camera if exists
                    if triggerbot.camera:
                        try:
                            triggerbot.camera.stop()
                        except:
                            pass
                    
                    # Force None and wait
                    triggerbot.camera = None
                    time.sleep(0.2)
                    
                    # Create new camera with updated region
                    triggerbot.camera = bettercam.create(output_idx=0, region=triggerbot.check_region)
                    
                    # Ensure FPS is an integer before passing it to camera.start
                    target_fps = int(triggerbot.fps)
                    triggerbot.camera.start(target_fps=target_fps)
                    debug_log(f"Camera recreated with new FOV - region: {triggerbot.check_region}, FPS: {target_fps}")
                    
                    # Reset the flag since we manually updated
                    triggerbot.region_changed = False
                    
                    self.log(f"FOV updated from {old_fov} to {new_fov} - camera region immediately changed")
                except Exception as e:
                    debug_log(f"Error during forced camera update: {e}")
                    traceback.print_exc()
                    self.log("Error updating camera - try restarting the TriggerBot")
            
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
            if not OPENCV_HAS_CUDA and self.config.get("use_gpu", False):
                self.config["use_gpu"] = False
                save_config(self.config)
                self.log("GPU acceleration disabled - OpenCV not compiled with CUDA support")
            
            # Use the average of min and max for backward compatibility
            avg_shooting_rate = (self.min_delay_spin.value() + self.max_delay_spin.value()) / 2
            
            # Create and start trigger bot thread
            self.trigger_bot_thread = TriggerBotThread(
                settings=self.config,
                fov=self.fov_spin.value(),
                shooting_rate=avg_shooting_rate,
                fps=self.config.get("fps", 200.0),
                hsv_range=self.config.get("hsv_range", [[30,125,150],[30,255,255]])
            )
            self.trigger_bot_thread.log_signal.connect(self.log)
            self.trigger_bot_thread.start()
            self.log("TriggerBot ACTIVATED")
        else:
            # Stop the trigger bot thread
            if self.trigger_bot_thread:
                self.trigger_bot_thread.stop()
                self.trigger_bot_thread.wait()
                self.trigger_bot_thread = None
            self.log("TriggerBot DEACTIVATED")

    def closeEvent(self, event):
        if self.trigger_bot_thread:
            self.trigger_bot_thread.stop()
            self.trigger_bot_thread.wait()
        if self.worker:
            shutdown_event.set()
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
        self.triggerbot_checkbox.setChecked(True)  # Changed to True to enable by default
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
        
        # GPU acceleration
        acceleration_group = QtWidgets.QGroupBox("GPU Acceleration")
        acceleration_layout = QtWidgets.QVBoxLayout(acceleration_group)
        
        # Create GPU status grid layout
        gpu_status_layout = QtWidgets.QGridLayout()
        
        # PyTorch status
        torch_label = QtWidgets.QLabel("PyTorch GPU:")
        torch_status = QtWidgets.QLabel()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch_status.setText(f"✅ Available - {torch.cuda.get_device_name(0)}")
            torch_status.setStyleSheet("color: #32CD32; font-weight: bold;")
        elif TORCH_AVAILABLE:
            torch_status.setText("⚠️ Installed but CUDA unavailable")
            torch_status.setStyleSheet("color: #FFA500; font-weight: bold;")
        else:
            torch_status.setText("❌ Not installed")
            torch_status.setStyleSheet("color: #FF6347; font-weight: bold;")
        
        gpu_status_layout.addWidget(torch_label, 0, 0)
        gpu_status_layout.addWidget(torch_status, 0, 1)
        
        # OpenCV status
        opencv_label = QtWidgets.QLabel("OpenCV GPU:")
        opencv_status = QtWidgets.QLabel()
        
        if OPENCV_HAS_CUDA and CUDA_AVAILABLE:
            opencv_status.setText(f"✅ Available")
            opencv_status.setStyleSheet("color: #32CD32; font-weight: bold;")
        elif OPENCV_HAS_CUDA:
            opencv_status.setText("⚠️ Compiled with CUDA but no devices found")
            opencv_status.setStyleSheet("color: #FFA500; font-weight: bold;")
        else:
            opencv_status.setText("❌ Not compiled with CUDA")
            opencv_status.setStyleSheet("color: #FF6347; font-weight: bold;")
        
        gpu_status_layout.addWidget(opencv_label, 1, 0)
        gpu_status_layout.addWidget(opencv_status, 1, 1)
        
        # Add to layout
        acceleration_layout.addLayout(gpu_status_layout)
        
        # Recommended acceleration method
        recommended_method = "CPU (no GPU acceleration available)"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            recommended_method = "PyTorch GPU"
        elif OPENCV_HAS_CUDA and CUDA_AVAILABLE:
            recommended_method = "OpenCV GPU"
        
        recommend_label = QtWidgets.QLabel(f"Recommended method: {recommended_method}")
        recommend_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        acceleration_layout.addWidget(recommend_label)
        
        # Add horizontal line separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        acceleration_layout.addWidget(line)
        
        # GPU usage toggle
        gpu_enabled = TORCH_AVAILABLE and torch.cuda.is_available() or (OPENCV_HAS_CUDA and CUDA_AVAILABLE)
        self.use_gpu_checkbox = QtWidgets.QCheckBox("Use GPU Acceleration")
        self.use_gpu_checkbox.setChecked(self.config.get("use_gpu", True) and gpu_enabled)
        self.use_gpu_checkbox.setToolTip("Enable GPU acceleration for faster color detection")
        
        if not gpu_enabled:
            self.use_gpu_checkbox.setChecked(False)
            self.use_gpu_checkbox.setEnabled(False)
            self.use_gpu_checkbox.setToolTip("GPU acceleration not available - CPU mode only")
            
        self.use_gpu_checkbox.stateChanged.connect(self.update_gpu_config)
        acceleration_layout.addWidget(self.use_gpu_checkbox)
        
        # Auto fallback toggle
        self.auto_fallback_checkbox = QtWidgets.QCheckBox("Auto Fallback to CPU if GPU Fails")
        self.auto_fallback_checkbox.setChecked(self.config.get("auto_fallback_to_cpu", True))
        self.auto_fallback_checkbox.setToolTip("Automatically switch to CPU mode if GPU detection fails multiple times")
        self.auto_fallback_checkbox.stateChanged.connect(self.update_fallback_config)
        acceleration_layout.addWidget(self.auto_fallback_checkbox)
        
        # Smart acceleration checkbox
        self.smart_accel_checkbox = QtWidgets.QCheckBox("Smart Acceleration (Auto-select best method)")
        self.smart_accel_checkbox.setChecked(self.config.get("smart_acceleration", True))
        self.smart_accel_checkbox.setToolTip("Automatically benchmark and select the fastest method")
        self.smart_accel_checkbox.stateChanged.connect(self.update_smart_acceleration)
        acceleration_layout.addWidget(self.smart_accel_checkbox)
        
        # Add test mode checkbox for debugging
        self.test_mode_checkbox = QtWidgets.QCheckBox("Enable Test Mode (Alternate GPU/CPU)")
        self.test_mode_checkbox.setChecked(self.config.get("test_mode", False))
        self.test_mode_checkbox.setToolTip("Alternate between GPU and CPU processing every few frames to compare performance")
        self.test_mode_checkbox.stateChanged.connect(self.update_test_mode)
        acceleration_layout.addWidget(self.test_mode_checkbox)
        
        # Buttons Row
        button_row = QtWidgets.QHBoxLayout()
        
        # Run benchmark button
        benchmark_button = QtWidgets.QPushButton("Run Performance Benchmark")
        benchmark_button.setToolTip("Run a benchmark to compare CPU vs GPU performance")
        benchmark_button.clicked.connect(self.run_benchmark)
        button_row.addWidget(benchmark_button)
        
        # Debug report button
        debug_button = QtWidgets.QPushButton("Generate Debug Report")
        debug_button.clicked.connect(self.generate_debug_report)
        debug_button.setToolTip("Create a report with system and CUDA information to help troubleshoot")
        button_row.addWidget(debug_button)
        
        acceleration_layout.addLayout(button_row)
        
        layout.addWidget(acceleration_group)
        
        # Add spacer
        layout.addStretch()
    
    def setup_settings_tab(self):
        layout = QtWidgets.QVBoxLayout(self.settings_tab)
        
        # Performance settings
        perf_group = QtWidgets.QGroupBox("Performance Settings")
        perf_layout = QtWidgets.QFormLayout(perf_group)
        
        # FOV setting
        self.fov_spin = QtWidgets.QDoubleSpinBox()
        self.fov_spin.setRange(1.0, 50.0)
        self.fov_spin.setValue(self.config.get("fov", 5.0))
        self.fov_spin.setSingleStep(0.5)
        self.fov_spin.setDecimals(1)
        self.fov_spin.valueChanged.connect(self.update_config)
        perf_layout.addRow("Detection FOV:", self.fov_spin)
        
        # Delay setting - replaced with Min/Max Delay
        delay_layout = QtWidgets.QHBoxLayout()
        
        # Min delay setting
        self.min_delay_spin = QtWidgets.QDoubleSpinBox()
        self.min_delay_spin.setRange(10.0, 500.0)
        self.min_delay_spin.setValue(self.config.get("min_shooting_rate", 50.0))
        self.min_delay_spin.setSingleStep(5.0)
        self.min_delay_spin.setDecimals(0)
        min_delay_label = QtWidgets.QLabel("Min:")
        
        # Max delay setting
        self.max_delay_spin = QtWidgets.QDoubleSpinBox()
        self.max_delay_spin.setRange(10.0, 500.0)
        self.max_delay_spin.setValue(self.config.get("max_shooting_rate", 80.0))
        self.max_delay_spin.setSingleStep(5.0)
        self.max_delay_spin.setDecimals(0)
        max_delay_label = QtWidgets.QLabel("Max:")
        
        # Connect value changed signals
        self.min_delay_spin.valueChanged.connect(self.update_config)
        self.max_delay_spin.valueChanged.connect(self.update_config)
        
        # Add to layout
        delay_layout.addWidget(min_delay_label)
        delay_layout.addWidget(self.min_delay_spin)
        delay_layout.addWidget(max_delay_label)
        delay_layout.addWidget(self.max_delay_spin)
        
        delay_container = QtWidgets.QWidget()
        delay_container.setLayout(delay_layout)
        perf_layout.addRow("Shooting Delay (ms):", delay_container)
        
        # FPS setting
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(30.0, 1000.0)
        self.fps_spin.setValue(self.config.get("fps", 200.0))
        self.fps_spin.setSingleStep(10.0)
        self.fps_spin.setDecimals(0)
        self.fps_spin.valueChanged.connect(self.update_config)
        perf_layout.addRow("Target FPS:", self.fps_spin)
        
        layout.addWidget(perf_group)
        
        # Click behavior
        click_group = QtWidgets.QGroupBox("Click Behavior")
        click_layout = QtWidgets.QVBoxLayout(click_group)
        
        # Random press duration
        duration_layout = QtWidgets.QHBoxLayout()
        self.press_duration_toggle = QtWidgets.QCheckBox("Enable Random Press Duration")
        self.press_duration_toggle.setChecked(self.config.get("enable_press_duration", True))
        self.press_duration_toggle.stateChanged.connect(self.on_press_duration_toggle)
        duration_layout.addWidget(self.press_duration_toggle)
        click_layout.addLayout(duration_layout)
        
        # Press duration settings
        press_duration_layout = QtWidgets.QFormLayout()
        self.press_duration_min_spin = QtWidgets.QDoubleSpinBox()
        self.press_duration_min_spin.setRange(0.01, 1.00)
        self.press_duration_min_spin.setValue(self.config.get("press_duration_min", 0.01))
        self.press_duration_min_spin.setSingleStep(0.01)
        self.press_duration_min_spin.setDecimals(2)
        self.press_duration_min_spin.valueChanged.connect(self.update_press_duration_min)
        press_duration_layout.addRow("Min Duration (s):", self.press_duration_min_spin)
        
        self.press_duration_max_spin = QtWidgets.QDoubleSpinBox()
        self.press_duration_max_spin.setRange(0.01, 1.00)
        self.press_duration_max_spin.setValue(self.config.get("press_duration_max", 0.05))
        self.press_duration_max_spin.setSingleStep(0.01)
        self.press_duration_max_spin.setDecimals(2)
        self.press_duration_max_spin.valueChanged.connect(self.update_press_duration_max)
        press_duration_layout.addRow("Max Duration (s):", self.press_duration_max_spin)
        
        click_layout.addLayout(press_duration_layout)
        
        # Block movement option
        self.block_movements_checkbox = QtWidgets.QCheckBox("Block Movement Keys While Shooting")
        self.block_movements_checkbox.setChecked(self.config.get("block_movements", False))
        self.block_movements_checkbox.stateChanged.connect(self.update_block_movements)
        self.block_movements_checkbox.setToolTip("When enabled, W/A/S/D keys will be temporarily released while shooting")
        click_layout.addWidget(self.block_movements_checkbox)
        
        layout.addWidget(click_group)
        
        # Add spacer
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
            
            # OpenCV info
            report.append("\n=== OpenCV Information ===")
            report.append(f"OpenCV Version: {cv2.__version__}")
            report.append(f"OpenCV CUDA Support Compiled: {OPENCV_HAS_CUDA}")
            # Get more detailed OpenCV build information
            try:
                cv_info = cv2.getBuildInformation()
                # Extract CUDA-related info from build information
                cuda_lines = [line for line in cv_info.split('\n') if 'CUDA' in line]
                for line in cuda_lines[:10]:  # Limit to avoid too much output
                    report.append(f"  {line.strip()}")
            except Exception as e:
                report.append(f"  Error getting OpenCV build info: {str(e)}")
            
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
            report.append(f"CUDA Available for Use: {CUDA_AVAILABLE}")
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
                if not OPENCV_HAS_CUDA:
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
            if OPENCV_HAS_CUDA:
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
                        
                    if benchmark_results['opencv_gpu_time'] < float('inf'):
                        self.log(f"OpenCV GPU time: {benchmark_results['opencv_gpu_time']:.2f}ms")
                    
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
                
            if benchmark_results['opencv_gpu_time'] < float('inf'):
                self.log(f"OpenCV GPU time: {benchmark_results['opencv_gpu_time']:.2f}ms")
            
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
        self.config["press_duration_min"] = value
        save_config(self.config)
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            self.log(f"Press duration minimum set to {value:.2f}s - applied immediately")

    def update_press_duration_max(self, value):
        """Update maximum press duration and apply immediately to running triggerbot"""
        self.config["press_duration_max"] = value
        save_config(self.config)
        
        # Update active triggerbot if running
        if self.trigger_bot_thread and self.trigger_bot_thread.isRunning():
            self.trigger_bot_thread.update_config(self.config)
            self.log(f"Press duration maximum set to {value:.2f}s - applied immediately")

def fast_benchmark(frame, iterations=10):
    """Run a quick benchmark to compare CPU vs GPU performance for real-time use"""
    try:
        # Make a copy of the frame to avoid modifying the original
        if frame is None:
            return {"best_method": "CPU", "ratio": 0}
            
        frame_copy = frame.copy()
        
        # Dictionary to store results
        results = {
            "cpu_time": 0.0001,  # Small epsilon to prevent division by zero
            "gpu_time": float('inf'),
            "ratio": 0,
            "best_method": "CPU"
        }
        
        # PyTorch GPU test (faster implementation for runtime)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Force clean CUDA memory
            torch.cuda.empty_cache()
            
            # Quick warmup
            test_frame = torch.from_numpy(frame_copy).cuda(non_blocking=True).float() / 255.0
            torch.cuda.synchronize()
            del test_frame
            
            # GPU benchmark
            gpu_start = time.time()
            
            for _ in range(iterations):
                # Simplified conversion to HSV and color detection
                frame_tensor = torch.from_numpy(frame_copy).cuda(non_blocking=True).float() / 255.0
                if len(frame_tensor.shape) == 3:
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                
                # Basic HSV conversion (simplified)
                max_val, _ = torch.max(frame_tensor, dim=0)
                min_val, _ = torch.min(frame_tensor, dim=0)
                
                # Just test with a simple operation to benchmark GPU transfer and basic compute
                diff = max_val - min_val
                result = torch.any(diff > 0.5).item()
                
                torch.cuda.synchronize()
            
            gpu_time = (time.time() - gpu_start) * 1000 / iterations
            results["gpu_time"] = max(gpu_time, 0.0001)  # Prevent zero
            
            # CPU test
            cpu_start = time.time()
            
            for _ in range(iterations):
                hsv = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2HSV)
                min_hsv = np.array([30, 125, 150], dtype=np.uint8)
                max_hsv = np.array([30, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, min_hsv, max_hsv)
                has_color = np.any(mask)
            
            cpu_time = (time.time() - cpu_start) * 1000 / iterations
            results["cpu_time"] = max(cpu_time, 0.0001)  # Prevent zero
            
            # Calculate results
            if gpu_time < float('inf'):
                if gpu_time > 0:
                    results["ratio"] = results["cpu_time"] / gpu_time
                    
                if results["ratio"] > 1:
                    results["best_method"] = "GPU"
                    debug_log(f"Quick benchmark: GPU is {results['ratio']:.2f}x faster ({gpu_time:.2f}ms vs {cpu_time:.2f}ms)")
                else:
                    results["best_method"] = "CPU"
                    slowdown = 1.0 / max(results["ratio"], 0.0001)
                    debug_log(f"Quick benchmark: GPU is {slowdown:.2f}x SLOWER ({gpu_time:.2f}ms vs {cpu_time:.2f}ms)")
            else:
                results["best_method"] = "CPU"
                debug_log("Quick benchmark: GPU not available")
        
        return results
    
    except Exception as e:
        debug_log(f"Error during quick benchmark: {str(e)}")
        return {"best_method": "CPU", "ratio": 0}

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
