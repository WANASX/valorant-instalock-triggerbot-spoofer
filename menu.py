import sys, os, time, random, ctypes, json, threading
from multiprocessing import Process, Queue
import cv2, numpy as np, pyautogui
import win32api, win32con
from PyQt5 import QtWidgets, QtCore, QtGui
import bettercam

# Global shutdown event used by the trigger bot process.
shutdown_event = threading.Event()
# UI-DO-NOT-OBFUSCATE-START
def save_config(config, filename='config.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print("Error saving config:", e)


# --- TriggerBot Shoot Simulation Process ---
def simulate_shoot(q):
    keybd_event = ctypes.windll.user32.keybd_event
    while not shutdown_event.is_set():
        try:
            signal_value = q.get(timeout=0.1)
        except Exception:
            continue
        if signal_value == "Shoot":
            press_duration = random.uniform(0.05, 0.15)
            keybd_event(0x01, 0, 0, 0)  # Always simulate left mouse button press
            time.sleep(press_duration)
            keybd_event(0x01, 0, 2, 0)  # Release left mouse button

# --- Color Detection Helper ---
def detect_color(frame, cmin, cmax):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, cmin, cmax)
        return np.any(mask)
    except cv2.error:
        return False


# --- Original TriggerBot Logic (wrapped for QThread use) ---
class Triggerbot:
    def __init__(self, q, keybind, fov, hsv_range, shooting_rate, fps):
        self.queue = q
        self.keybind = keybind
        self.shooting_rate = shooting_rate / 1000.0  # Convert ms to seconds.
        self.fps = int(fps)
        self.fov = int(fov)
        self.last_shot_time = 0
        
        user32 = ctypes.windll.user32
        self.WIDTH, self.HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        self.check_region = (
            max(0, center_x - self.fov),
            max(0, center_y - self.fov),
            min(self.WIDTH, center_x + self.fov),
            min(self.HEIGHT, center_y + self.fov)
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
# --- QThread Wrapper for TriggerBot ---
class TriggerBotThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    def __init__(self, keybind, fov, shooting_rate, fps, hsv_range, parent=None):
        super().__init__(parent)
        self.keybind = keybind
        self.fov = fov
        self.shooting_rate = shooting_rate
        self.fps = fps
        self.hsv_range = hsv_range
        self.triggerbot = None
        self.shoot_queue = Queue()
        self.shoot_process = None
    
    def run(self):
        # Start the shoot simulation process (always using left mouse button).
        self.shoot_process = Process(target=simulate_shoot, args=(self.shoot_queue,))
        self.shoot_process.start()
        # Create and run the trigger bot.
        self.triggerbot = Triggerbot(self.shoot_queue, self.keybind, self.fov, self.hsv_range, self.shooting_rate, self.fps)
        self.triggerbot.run()
    
    def stop(self):
        shutdown_event.set()
        if self.shoot_process:
            self.shoot_process.terminate()
            self.shoot_process.join()
        if self.triggerbot and hasattr(self.triggerbot, 'camera'):
            self.triggerbot.camera.stop()
        self.terminate()


# --- QThread for Scanning (for Agent and Lock images) ---
class ScanningWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()
    def __init__(self, agent_image_path, parent=None):
        super().__init__(parent)
        self.agent_image_path = agent_image_path
    def run(self):
        session_start = time.time()
        width, height = pyautogui.size()
        agent_region = (0, 0, width//2, height)
        lock_region = (0, height//2, width, height//2)
        self.log_signal.emit("Scanning for agent image...")
        found_agent = None
        timeout = 180  # 3 minutes timeout.
        while time.time() - session_start < timeout:
            try:
                pos = pyautogui.locateOnScreen(self.agent_image_path, region=agent_region, confidence=0.75, grayscale=True, minSearchTime=0.15)
            except Exception:
                pos = None
            if pos:
                found_agent = pos
                break
            time.sleep(0.3)
        if found_agent:
            center_x, center_y = pyautogui.center(found_agent)
            elapsed = time.time() - session_start
            self.log_signal.emit(f"✅ Agent found at ({center_x}, {center_y}) in {elapsed:.2f}s")
            human_move(int(center_x), int(center_y))
            natural_click(int(center_x), int(center_y))
            self.log_signal.emit("Scanning for Lock.png...")
            lock_path = os.path.join(os.getcwd(), "Lock.png")
            try:
                pos_lock = pyautogui.locateOnScreen(lock_path, region=lock_region, confidence=0.75, grayscale=True, minSearchTime=0.15)
            except Exception:
                pos_lock = None
            if pos_lock:
                lx, ly = pyautogui.center(pos_lock)
                self.log_signal.emit(f"✅ Lock found at ({lx}, {ly})")
                human_move(int(lx), int(ly))
                natural_click(int(lx), int(ly))
            else:
                self.log_signal.emit("❌ Lock.png not found.")
        else:
            self.log_signal.emit("❌ Agent image not found within 3 minutes.")
        self.finished_signal.emit()


# --- Movement and Clicking Helpers ---
def human_move(x, y):
    user32 = ctypes.windll.user32
    try:
        start_x, start_y = win32api.GetCursorPos()
        steps = random.randint(5, 12)
        cp1 = (start_x + (x - start_x)*random.uniform(0.1, 0.5),
               start_y + (y - start_y)*random.uniform(0.1, 0.4))
        cp2 = (start_x + (x - start_x)*random.uniform(0.5, 0.9),
               start_y + (y - start_y)*random.uniform(0.6, 1.0))
        for t in (i/steps for i in range(0, steps+1)):
            xt = (1-t)**3 * start_x + 3*(1-t)**2*t * cp1[0] + 3*(1-t)*t**2 * cp2[0] + t**3 * x
            yt = (1-t)**3 * start_y + 3*(1-t)**2*t * cp1[1] + 3*(1-t)*t**2 * cp2[1] + t**3 * y
            user32.SetCursorPos(int(xt), int(yt))
    except Exception as e:
        user32.SetCursorPos(x, y)
        print("Movement fallback:", str(e)[:50])

def perform_click(x, y, press_time):
    user32 = ctypes.windll.user32
    try:
        user32.SetCursorPos(x, y)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(press_time)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        user32.SetCursorPos(x+random.randint(-2,2), y+random.randint(-2,2))
    except Exception as e:
        print("Error in perform_click:", str(e))

def natural_click(x, y):
    try:
        final_x = x + random.randint(-5, 5)
        final_y = y + random.randint(-5, 5)
        press_time = 0.00001
        p = Process(target=perform_click, args=(final_x, final_y, press_time))
        p.start()
        p.join()
        time.sleep(0.001)
    except Exception as e:
        print("Click error:", str(e)[:50])


# --- MainWindow and UI ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.key_map = {
            "Alt": 164,
            "BROKEN1": 160,
            "BROKEN2": 88,
            "BROKEN3": 67,
            "BROKEN4": 86,
            "BROKEN5": 90,
            "BROKEN6": 66,
            "BROKEN7": 81,
            "BROKEN8": 82,
            "BROKEN9": 20,
        }


        self.config = load_config()
        
        self.app_titles = [
            "Telegram", "WhatsApp", "Discord", "Skype", "Slack", "Zoom", "Signal", "Microsoft Teams", 
            "Google Meet", "Viber", "Facebook Messenger", "WeChat", "Line", "Kik", "Snapchat", "Instagram", 
            "Twitter (X)", "Facebook", "LinkedIn", "Reddit", "TikTok", "Clubhouse", "Mastodon", "Threads", 
            "BeReal", "Spotify", "Apple Music", "YouTube", "Netflix", "Hulu", "Disney+", "Amazon Prime Video", 
            "HBO Max", "Twitch", "SoundCloud", "Deezer", "Pandora", "Tidal", "Google Drive", "Google Docs", 
            "Evernote", "Notion", "Trello", "Asana", "Monday.com", "ClickUp", "Todoist", "OneNote", "Dropbox", 
            "PayPal", "Venmo", "Cash App", "Zelle", "Google Pay", "Apple Pay", "Stripe", "Robinhood", "Revolut", 
            "Wise", "Amazon", "eBay", "Etsy", "Walmart", "AliExpress", "Shopify", "Temu", "Google Maps", 
            "Waze", "Uber", "Lyft", "Airbnb", "Booking.com", "Skyscanner", "MyFitnessPal", "Strava", "Fitbit", 
            "Calm", "Headspace"
        ]

        self.setWindowTitle(random.choice(self.app_titles))
        self.setFixedSize(500, 650)
        self.scanning_in_progress = False

        if not ctypes.windll.shell32.IsUserAnAdmin():
            QtWidgets.QMessageBox.critical(self, "Administrator Required", "Run this script as Administrator!")
            sys.exit(1)

        self.create_ui()
        self.agents_dir = os.path.join(os.getcwd(), "Agents")
        self.load_agent_images()
        self.hotkey_timer = QtCore.QTimer(self)
        self.hotkey_timer.timeout.connect(self.check_hotkey)
        self.hotkey_timer.start(50)
        self.trigger_bot_thread = None

    def create_ui(self):
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setSpacing(10)
        
        title_label = QtWidgets.QLabel("GamerFun Valo 1.1")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Agent selection dropdown.
        agent_layout = QtWidgets.QHBoxLayout()
        agent_label = QtWidgets.QLabel("Select Agent Image:")
        self.agent_combo = QtWidgets.QComboBox()
        agent_layout.addWidget(agent_label)
        agent_layout.addWidget(self.agent_combo)
        layout.addLayout(agent_layout)
        
        info_label = QtWidgets.QLabel("Press F5 to scan until target is found (timeout = 3 min).")
        layout.addWidget(info_label)
        
        # TriggerBot Enable Checkbox and Activation Key dropdown.
        trigger_layout = QtWidgets.QHBoxLayout()
        self.triggerbot_checkbox = QtWidgets.QCheckBox("Enable TriggerBot")
        trigger_layout.addWidget(self.triggerbot_checkbox)
        key_label = QtWidgets.QLabel("Trigger Bot Key:")
        self.trigger_key_combo = QtWidgets.QComboBox()
        self.trigger_key_combo.setEditable(True)
        self.trigger_key_combo.addItems(list(self.key_map.keys()))
        inv_key_map = {v: k for k, v in self.key_map.items()}
        initial_key = self.config.get("keybind", 164)
        self.trigger_key_combo.setCurrentText(inv_key_map.get(initial_key, str(initial_key)))
        trigger_layout.addWidget(key_label)
        trigger_layout.addWidget(self.trigger_key_combo)
        layout.addLayout(trigger_layout)
        
        # Trigger Bot Config group: FOV and Delay.
        config_group = QtWidgets.QGroupBox("Trigger Bot Config")
        form = QtWidgets.QFormLayout()
        self.fov_spin = QtWidgets.QDoubleSpinBox()
        self.fov_spin.setRange(1.0, 50.0)
        self.fov_spin.setValue(self.config.get("fov", 5.0))
        self.fov_spin.setSingleStep(0.1)
        form.addRow("FOV:", self.fov_spin)
        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(10.0, 500.0)
        self.delay_spin.setValue(self.config.get("shooting_rate", 65.0))
        self.delay_spin.setSingleStep(1.0)
        form.addRow("Delay (ms):", self.delay_spin)
        config_group.setLayout(form)
        layout.addWidget(config_group)
        
        # Log display.
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #C0C0C0; font-family: Segoe UI, sans-serif; }
            QComboBox, QTextEdit, QLabel, QDoubleSpinBox { background-color: #1e1e2e; border: 1px solid #2e2e2e; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1e1e2e; selection-background-color: #32CD32; }
            QLabel { color: #32CD32; }
        """)
        
        self.triggerbot_checkbox.stateChanged.connect(self.toggle_triggerbot)
        self.trigger_key_combo.currentIndexChanged.connect(self.update_keybind)
        self.trigger_key_combo.lineEdit().editingFinished.connect(self.update_keybind)
        self.fov_spin.valueChanged.connect(self.update_config)
        self.delay_spin.valueChanged.connect(self.update_config)
        
    def load_agent_images(self):
        if not os.path.exists(self.agents_dir):
            self.log("Agents folder not found!")
            return
        imgs = [f for f in os.listdir(self.agents_dir) if f.lower().endswith(".png")]
        if imgs:
            self.agent_combo.addItems(imgs)
            self.log(f"Loaded {len(imgs)} agent image(s).")
        else:
            self.log("No PNG images found in the Agents folder.")
            
    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.append(timestamp + message)
        
    def check_hotkey(self):
        if win32api.GetAsyncKeyState(win32con.VK_F5) < 0:
            if not self.scanning_in_progress:
                self.start_scanning_session()
            else:
                self.log("Scan session already in progress.")
        
    def start_scanning_session(self):
        agent = self.agent_combo.currentText()
        if not agent:
            self.log("No agent image selected!")
            return
        path = os.path.join(self.agents_dir, agent)
        self.log(f"Starting scan for agent: {agent}")
        self.scanning_in_progress = True
        self.worker = ScanningWorker(path)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.session_finished)
        self.worker.start()
        
    def session_finished(self):
        self.scanning_in_progress = False
        self.log("Scan session ended.")
        
    def toggle_triggerbot(self, state):
        if state == QtCore.Qt.Checked:
            self.update_keybind()
            self.config["fov"] = self.fov_spin.value()
            self.config["shooting_rate"] = self.delay_spin.value()
            save_config(self.config)
            self.log("TriggerBot enabled!")
            self.trigger_bot_thread = TriggerBotThread(
                keybind=self.config.get("keybind", 164),
                fov=self.fov_spin.value(),
                shooting_rate=self.delay_spin.value(),
                fps=self.config.get("fps", 200.0),
                hsv_range=self.config.get("hsv_range", [[30,125,150],[30,255,255]])
            )
            self.trigger_bot_thread.log_signal.connect(self.log)
            self.trigger_bot_thread.start()
        else:
            if self.trigger_bot_thread:
                self.trigger_bot_thread.stop()
                self.trigger_bot_thread.wait()
                self.trigger_bot_thread = None
            self.log("TriggerBot disabled!")
            
    def update_keybind(self):
        text = self.trigger_key_combo.currentText().strip()
        if text in self.key_map:
            code = self.key_map[text]
        else:
            try:
                code = int(text)
            except ValueError:
                code = ord(text.upper()[0]) if text else 164
        self.config["keybind"] = code
        save_config(self.config)
        
    def update_config(self):
        self.config["fov"] = self.fov_spin.value()
        self.config["shooting_rate"] = self.delay_spin.value()
        save_config(self.config)
        
    def closeEvent(self, event):
        if self.trigger_bot_thread:
            self.trigger_bot_thread.stop()
            self.trigger_bot_thread.wait()
        event.accept()


# --- Configuration Functions ---
def load_config(filename='config.json'):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading config:", e)
    return {
        "fov": 5.0,
        "keybind": 164,         # Default trigger activation key (e.g. Alt key code)
        "shooting_rate": 65.0,    # Delay in ms between shots
        "fps": 200.0,
        "hsv_range": [[30, 125, 150], [30, 255, 255]]
    }

def main():

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
# UI-DO-NOT-OBFUSCATE-END