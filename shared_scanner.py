import numpy as np
import mss
import cv2
import torch
from collections import defaultdict
import threading
import time

class SharedScanner:
    """
    Unified, optimized scanner for screen capture and color detection.
    Supports both CPU (OpenCV) and GPU (PyTorch, fp16) color detection.
    Deeply optimized for minimal allocations and maximal throughput.
    """
    def __init__(self, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # Buffer cache: (height, width) -> (pinned numpy array, cuda tensor)
        self._gpu_buffers = {}
        self._last_shape = None

    def _get_gpu_buffers(self, h, w):
        """
        Get or create pinned numpy and CUDA fp16 tensor buffers for a given region size.
        Returns: (pinned_np, cuda_tensor)
        """
        key = (h, w)
        if key not in self._gpu_buffers:
            # Pinned memory numpy array for fast CPU->GPU transfer
            pinned_np = np.empty((h, w, 3), dtype=np.float16)
            pinned_np = np.ascontiguousarray(pinned_np)
            # Preallocate CUDA tensor
            cuda_tensor = torch.empty((h * w, 3), dtype=torch.float16, device=self.device)
            self._gpu_buffers[key] = (pinned_np, cuda_tensor)
        return self._gpu_buffers[key]

    def grab_screen(self, region):
        """
        Capture a region of the screen as a numpy array (BGRA).
        region: dict with keys 'left', 'top', 'width', 'height'
        Returns: np.ndarray (H, W, 4)
        """
        with mss.mss() as sct:
            img = np.asarray(sct.grab(region))
        return img

    def detect_color_cpu(self, img, lower_bgr, upper_bgr):
        """
        Detect if any pixel in img is within the BGR color range (OpenCV, CPU).
        img: np.ndarray (H, W, 3 or 4)
        lower_bgr, upper_bgr: tuple/list of 3 ints (B, G, R)
        Returns: bool
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]
        mask = cv2.inRange(img, np.array(lower_bgr, dtype=np.uint8), np.array(upper_bgr, dtype=np.uint8))
        return np.any(mask)

    def detect_color_hsv_cpu(self, img, lower_hsv, upper_hsv):
        """
        Detect if any pixel in img is within the HSV color range (OpenCV, CPU).
        img: np.ndarray (H, W, 3 or 4)
        lower_hsv, upper_hsv: tuple/list of 3 ints (H, S, V)
        Returns: bool
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8))
        return np.any(mask)

    def detect_color_gpu(self, img, target_bgr, tolerance):
        """
        Detect if any pixel in img is within tolerance of target_bgr (PyTorch, GPU, fp16).
        Uses preallocated pinned/cuda buffers and vectorized ops.
        img: np.ndarray (H, W, 3 or 4)
        target_bgr: tuple/list of 3 floats (B, G, R)
        tolerance: float (Euclidean distance in BGR space)
        Returns: bool
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available for GPU detection.")
        if img.shape[2] == 4:
            img = img[:, :, :3]
        h, w, _ = img.shape
        pinned_np, cuda_tensor = self._get_gpu_buffers(h, w)
        # Copy to pinned buffer (convert to float16 in-place)
        np.copyto(pinned_np, img.astype(np.float16, copy=False))
        # Transfer to CUDA tensor (async, non-blocking)
        cuda_tensor.copy_(torch.from_numpy(pinned_np).reshape(-1, 3), non_blocking=True)
        # Vectorized color distance
        target = torch.tensor(target_bgr, dtype=torch.float16, device=self.device)
        dist = torch.sum((cuda_tensor - target) ** 2, dim=1)
        matches = torch.sum(dist <= tolerance ** 2)
        torch.cuda.synchronize()  # Only sync after detection
        return matches.item() > 0

    def detect_color_hsv_gpu(self, img, lower_hsv, upper_hsv):
        """
        Detect if any pixel in img is within the HSV color range (PyTorch, GPU, fp16).
        Uses preallocated pinned/cuda buffers and vectorized ops.
        img: np.ndarray (H, W, 3 or 4)
        lower_hsv, upper_hsv: tuple/list of 3 floats (H, S, V)
        Returns: bool
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available for GPU detection.")
        if img.shape[2] == 4:
            img = img[:, :, :3]
        h, w, _ = img.shape
        pinned_np, cuda_tensor = self._get_gpu_buffers(h, w)
        np.copyto(pinned_np, img.astype(np.float16, copy=False))
        cuda_tensor.copy_(torch.from_numpy(pinned_np).reshape(-1, 3), non_blocking=True)
        # Vectorized BGR->HSV conversion (approximate, on GPU)
        b, g, r = cuda_tensor[:, 0] / 255.0, cuda_tensor[:, 1] / 255.0, cuda_tensor[:, 2] / 255.0
        maxc = torch.max(torch.stack([b, g, r], dim=1), dim=1).values
        minc = torch.min(torch.stack([b, g, r], dim=1), dim=1).values
        v = maxc
        s = torch.where(maxc != 0, (maxc - minc) / maxc, torch.zeros_like(maxc))
        h = torch.zeros_like(maxc)
        mask = (maxc == r) & (maxc != minc)
        h[mask] = (60 * (g[mask] - b[mask]) / (maxc[mask] - minc[mask]) + 360) % 360
        mask = (maxc == g) & (maxc != minc)
        h[mask] = (60 * (b[mask] - r[mask]) / (maxc[mask] - minc[mask]) + 120)
        mask = (maxc == b) & (maxc != minc)
        h[mask] = (60 * (r[mask] - g[mask]) / (maxc[mask] - minc[mask]) + 240)
        h = h / 2  # OpenCV uses [0, 179]
        s = s * 255
        v = v * 255
        hsv = torch.stack([h, s, v], dim=1)
        lower = torch.tensor(lower_hsv, dtype=torch.float16, device=self.device)
        upper = torch.tensor(upper_hsv, dtype=torch.float16, device=self.device)
        mask = torch.all((hsv >= lower) & (hsv <= upper), dim=1)
        torch.cuda.synchronize()
        return torch.any(mask).item()

    def scan(self, region, mode, color_range, tolerance=None, color_space='bgr'):
        """
        Unified scan API for both CPU and GPU, BGR or HSV.
        region: dict for mss
        mode: 'cpu' or 'gpu'
        color_range: (lower, upper) for BGR/HSV or (target, tolerance) for BGR+tol
        tolerance: float, only for BGR+tol
        color_space: 'bgr' or 'hsv'
        Returns: bool
        """
        img = self.grab_screen(region)
        if mode == 'cpu':
            if color_space == 'bgr':
                return self.detect_color_cpu(img, *color_range)
            else:
                return self.detect_color_hsv_cpu(img, *color_range)
        elif mode == 'gpu':
            if color_space == 'bgr':
                return self.detect_color_gpu(img, *color_range, tolerance)
            else:
                return self.detect_color_hsv_gpu(img, *color_range)
        else:
            raise ValueError("mode must be 'cpu' or 'gpu'")

class ScanningManager:
    """
    ScanningManager is a singleton/service that manages all screen scanning and detection for subscribers (e.g., TriggerBot, AimLock).
    - Owns a SharedScanner instance and a scanning thread.
    - Supports multiple regions per frame.
    - Notifies subscribers (callbacks) with detection results for their region(s).
    - Thread-safe and efficient.
    
    Usage:
        manager = ScanningManager.get_instance()
        manager.register_subscriber(region_name, region_dict, scan_params, callback)
        manager.start()
        ...
        manager.stop()
    """
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, fps=240):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(fps=fps)
            return cls._instance

    def __init__(self, fps=240):
        self.scanner = SharedScanner()
        self.fps = fps
        self.regions = {}  # region_name -> region_dict
        self.scan_params = {}  # region_name -> dict (mode, color_range, tolerance, color_space)
        self.subscribers = defaultdict(list)  # region_name -> list of callbacks
        self._lock = threading.RLock()
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    def register_subscriber(self, region_name, region_dict, scan_params, callback):
        """
        Register a subscriber for a region.
        - region_name: str (unique key for the region)
        - region_dict: dict (mss region dict)
        - scan_params: dict (mode, color_range, tolerance, color_space)
        - callback: function(result: bool, frame: np.ndarray, region_name: str)
        """
        with self._lock:
            self.regions[region_name] = region_dict
            self.scan_params[region_name] = scan_params
            if callback not in self.subscribers[region_name]:
                self.subscribers[region_name].append(callback)

    def unregister_subscriber(self, region_name, callback=None):
        """
        Unregister a subscriber or all subscribers for a region.
        If callback is None, remove all subscribers for the region.
        """
        with self._lock:
            if region_name in self.subscribers:
                if callback:
                    if callback in self.subscribers[region_name]:
                        self.subscribers[region_name].remove(callback)
                else:
                    self.subscribers[region_name] = []
            # Optionally remove region if no subscribers left
            if not self.subscribers[region_name]:
                self.regions.pop(region_name, None)
                self.scan_params.pop(region_name, None)
                self.subscribers.pop(region_name, None)
        # --- Improvement: Stop scanning thread if no subscribers left ---
        with self._lock:
            if not self.subscribers:
                if self._running:
                    print("[ScanningManager] No subscribers left, stopping scanning thread.")
                self.stop()

    def update_region(self, region_name, region_dict=None, scan_params=None):
        """
        Update the region dict and/or scan params for a region.
        """
        with self._lock:
            if region_dict is not None:
                self.regions[region_name] = region_dict
            if scan_params is not None:
                self.scan_params[region_name] = scan_params

    def start(self):
        """Start the scanning thread."""
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._scan_loop, name="ScanningManagerThread", daemon=True)
            self._running = True
            self._thread.start()

    def stop(self):
        """Stop the scanning thread."""
        with self._lock:
            self._stop_event.set()
            self._running = False
            if self._thread:
                self._thread.join(timeout=2)
                self._thread = None

    def _scan_loop(self):
        """
        For each region, capture once, run detection once, notify all subscribers.
        Runs at self.fps (configurable).
        """
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            with self._lock:
                regions = dict(self.regions)
                scan_params = dict(self.scan_params)
                subscribers = {k: list(v) for k, v in self.subscribers.items()}
            for region_name, region_dict in regions.items():
                params = scan_params.get(region_name, {})
                try:
                    frame = self.scanner.grab_screen(region_dict)
                    result = self.scanner.scan(
                        region_dict,
                        mode=params.get('mode', 'cpu'),
                        color_range=params.get('color_range', ((0,0,0),(255,255,255))),
                        tolerance=params.get('tolerance', None),
                        color_space=params.get('color_space', 'bgr')
                    )
                    for callback in subscribers.get(region_name, []):
                        try:
                            callback(result, frame, region_name)
                        except Exception as cb_exc:
                            print(f"[ScanningManager] Callback error for region '{region_name}': {cb_exc}")
                except Exception as e:
                    print(f"[ScanningManager] Scan error for region '{region_name}': {e}")
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, 1.0/self.fps - elapsed)
            time.sleep(sleep_time)

    def is_running(self):
        """Return True if the scanning thread is running."""
        with self._lock:
            return self._running 