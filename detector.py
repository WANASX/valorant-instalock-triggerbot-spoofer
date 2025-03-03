import sys
import os
import time
import random
import ctypes
import pyautogui
import win32api
import win32con
import multiprocessing

from PyQt5 import QtWidgets, QtCore, QtGui

# ======== ENHANCED CONFIGURATION ========
MIN_CONFIDENCE = 0.75
JITTER_RANGE = 5
TIMEOUT_SECONDS = 3 * 60  # 3 minutes timeout for scanning
# =======================================

user32 = ctypes.windll.user32

# ---------- Improved Movement Function ----------
def human_move(x, y):
    """Enhanced Bézier movement with fallback (improved speed)"""
    try:
        start_x, start_y = win32api.GetCursorPos()
        steps = random.randint(5, 12)  # Fewer steps for faster movement
    
        # Dynamic control points
        cp1 = (
            start_x + (x - start_x) * random.uniform(0.1, 0.5),
            start_y + (y - start_y) * random.uniform(0.1, 0.4)
        )
        cp2 = (
            start_x + (x - start_x) * random.uniform(0.5, 0.9),
            start_y + (y - start_y) * random.uniform(0.6, 1.0)
        )
        for t in (i/steps for i in range(0, steps+1)):
            xt = (1-t)**3 * start_x + 3*(1-t)**2*t * cp1[0] + 3*(1-t)*t**2 * cp2[0] + t**3 * x
            yt = (1-t)**3 * start_y + 3*(1-t)**2*t * cp1[1] + 3*(1-t)*t**2 * cp2[1] + t**3 * y
            user32.SetCursorPos(int(xt), int(yt))
            
    except Exception as e:
        user32.SetCursorPos(x, y)
        print(f"Movement fallback: {str(e)[:50]}")

# ---------- Improved Per-Click Process Logic ----------
def perform_click(x, y, press_time):
    """Performs a single click using Windows API calls.
       Runs in its own process so each click comes from a different app context."""
    try:
        user32.SetCursorPos(x, y)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(press_time)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        # Micro-movement after click
        user32.SetCursorPos(x + random.randint(-2, 2), y + random.randint(-2, 2))
    except Exception as e:
        print(f"Error in perform_click: {e}")

def natural_click(x, y):
    """Enhanced click with triple verification.
       For each click, a separate process is spawned so that every click appears to come from a different app."""
    try:
        for _ in range(1):  # Triple click attempt
            # Randomize final position
            final_x = x + random.randint(-JITTER_RANGE, JITTER_RANGE)
            final_y = y + random.randint(-JITTER_RANGE, JITTER_RANGE)
            press_time = 0.00001
            # Spawn a new process for each click
            p = multiprocessing.Process(target=perform_click, args=(final_x, final_y, press_time))
            p.start()
            p.join()  # Wait for this click process to complete
            time.sleep(0.001)  # Minimal delay between clicks
    except Exception as e:
        print(f"Click error: {str(e)[:50]}")

# ---------- Improved Adaptive Detection Function ----------
def adaptive_detection_for(image_path, region, initial_confidence=MIN_CONFIDENCE):
    """
    Searches for image_path on-screen within the given region.
    Uses three attempts with confidence adjustments.
    """
    global MIN_CONFIDENCE
    try:
        for _ in range(3):
            try:
                pos = pyautogui.locateOnScreen(
                    image_path,
                    region=region,
                    confidence=MIN_CONFIDENCE,
                    grayscale=True,
                    minSearchTime=0.15  # Reduced search time for speed
                )
                if pos:
                    MIN_CONFIDENCE = max(0.7, MIN_CONFIDENCE - 0.01)
                    return pos
                else:
                    MIN_CONFIDENCE = min(0.85, MIN_CONFIDENCE + 0.02)
            except Exception:
                continue
        MIN_CONFIDENCE = 0.75
        return None
    except Exception:
        return None

# ---------- Worker Thread for Scanning ----------
class ScanningWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()
    
    def __init__(self, agent_image_path, parent=None):
        super().__init__(parent)
        self.agent_image_path = agent_image_path  # Full path to the selected agent image

    def run(self):
        global MIN_CONFIDENCE
        session_start = time.time()
        width, height = pyautogui.size()
        # Define scanning regions:
        agent_region = (0, 0, width // 2, height)          # left half for agent image
        lock_region = (0, height // 2, width, height // 2)    # bottom half for Lock.png

        self.log_signal.emit("Scanning for agent image...")
        found_agent = None
        # Poll frequently for a fast response
        while time.time() - session_start < TIMEOUT_SECONDS:
            found_agent = adaptive_detection_for(self.agent_image_path, agent_region)
            if found_agent:
                break
            time.sleep(0.3)
        
        if found_agent:
            center_x, center_y = pyautogui.center(found_agent)
            elapsed = time.time() - session_start
            self.log_signal.emit(f"✅ Agent found at ({center_x}, {center_y}) in {elapsed:.2f}s")
            human_move(int(center_x), int(center_y))
            natural_click(int(center_x), int(center_y))
            MIN_CONFIDENCE = 0.75
            self.log_signal.emit("Scanning for Lock.png...")
            lock_image_path = os.path.join(os.getcwd(), "Lock.png")
            found_lock = adaptive_detection_for(lock_image_path, lock_region)
            if found_lock:
                lock_center_x, lock_center_y = pyautogui.center(found_lock)
                self.log_signal.emit(f"✅ Lock found at ({lock_center_x}, {lock_center_y})")
                human_move(int(lock_center_x), int(lock_center_y))
                natural_click(int(lock_center_x), int(lock_center_y))
            else:
                self.log_signal.emit("❌ Lock.png not found.")
        else:
            self.log_signal.emit("❌ Agent image not found within 3 minutes.")
        MIN_CONFIDENCE = 0.75
        self.finished_signal.emit()

# ---------- Main Window UI ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telegram")
        self.setFixedSize(500, 400)
        self.scanning_in_progress = False
        self.prev_f5_state = False  # For F5 edge detection

        # Ensure running as Administrator
        if not ctypes.windll.shell32.IsUserAnAdmin():
            QtWidgets.QMessageBox.critical(
                self, "Administrator Required", "Run this script as Administrator!"
            )
            sys.exit(1)

        self.create_ui()
        self.agents_dir = os.path.join(os.getcwd(), "Agents")
        self.load_agent_images()

        self.hotkey_timer = QtCore.QTimer(self)
        self.hotkey_timer.timeout.connect(self.check_hotkey)
        self.hotkey_timer.start(50)

    def create_ui(self):
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setSpacing(10)

        title_label = QtWidgets.QLabel("🔥 GamerFun Valo Menu 0.1")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Dropdown for agent image selection
        dropdown_layout = QtWidgets.QHBoxLayout()
        agent_label = QtWidgets.QLabel("Select Agent Image:")
        self.agent_combo = QtWidgets.QComboBox()
        dropdown_layout.addWidget(agent_label)
        dropdown_layout.addWidget(self.agent_combo)
        layout.addLayout(dropdown_layout)

        info_label = QtWidgets.QLabel("Press F5 to scan until target is found (timeout = 3 min).")
        layout.addWidget(info_label)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #C0C0C0;
                font-family: Segoe UI, sans-serif;
            }
            QComboBox, QTextEdit, QLabel {
                background-color: #1e1e1e;
                border: 1px solid #2e2e2e;
                color: #C0C0C0;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1e1e1e; selection-background-color: #32CD32; }
            QLabel { color: #32CD32; }
        """)

    def load_agent_images(self):
        if not os.path.exists(self.agents_dir):
            self.log("Agents folder not found!")
            return
        images = [f for f in os.listdir(self.agents_dir) if f.lower().endswith(".png")]
        if images:
            self.agent_combo.addItems(images)
            self.log(f"Loaded {len(images)} agent image(s).")
        else:
            self.log("No PNG images found in the Agents folder.")

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.append(timestamp + message)

    def check_hotkey(self):
        f5_pressed = (win32api.GetAsyncKeyState(win32con.VK_F5) < 0)
        if f5_pressed and not self.prev_f5_state:
            if not self.scanning_in_progress:
                self.start_scanning_session()
            else:
                self.log("Scan session already in progress.")
        self.prev_f5_state = f5_pressed

    def start_scanning_session(self):
        selected_agent = self.agent_combo.currentText()
        if not selected_agent:
            self.log("No agent image selected!")
            return
        agent_image_path = os.path.join(self.agents_dir, selected_agent)
        self.log(f"Starting scan for agent: {selected_agent}")
        self.scanning_in_progress = True
        self.worker = ScanningWorker(agent_image_path)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.session_finished)
        self.worker.start()

    def session_finished(self):
        self.scanning_in_progress = False
        self.log("Scan session ended.")

# ---------- Main Entry Point ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    exit_code = app.exec_()
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
