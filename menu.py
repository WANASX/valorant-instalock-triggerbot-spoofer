import sys, os, time, random, ctypes, json, threading
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2, numpy as np, pyautogui
import win32api, win32con
from PyQt5 import QtWidgets, QtCore, QtGui
import bettercam
from tempfile import gettempdir
from PIL import Image

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
        "fps": 200.0,
        "hsv_range": [[30, 125, 150], [30, 255, 255]],
        "trigger_mode": "hold",  # or "toggle"
        "press_duration_min": 0.01,  # Minimum press duration in seconds
        "press_duration_max": 0.05,  # Maximum press duration in seconds
        "enable_press_duration": True,  # Toggle for using random press duration
        "block_movements": False   # If True, block (W, A, S, D) while shooting
    }

# ------------------- TriggerBot Logic -------------------

def simulate_shoot(q, config):
    keybd_event = ctypes.windll.user32.keybd_event
    while not shutdown_event.is_set():
        try:
            signal_value = q.get(timeout=0.1)
        except Exception:
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

def detect_color(frame, cmin, cmax):
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, cmin, cmax)
        return np.any(mask)
    except cv2.error:
        return False

class Triggerbot:
    def __init__(self, q, settings, fov, hsv_range, shooting_rate, fps):
        self.queue = q
        self.settings = settings  # shared config dictionary
        self.shooting_rate = shooting_rate / 1000.0
        self.fps = int(fps)
        self.fov = int(fov)
        self.last_shot_time = 0
        self.toggled_active = False  # used only for toggle mode
        self.last_key_state = False

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
            current_key = self.settings.get("keybind", 164)
            mode = self.settings.get("trigger_mode", "hold")
            if mode == "hold":
                active = (win32api.GetAsyncKeyState(current_key) < 0)
            else:
                current_state = (win32api.GetAsyncKeyState(current_key) < 0)
                if current_state and not self.last_key_state:
                    self.toggled_active = not self.toggled_active
                    time.sleep(0.3)  # debounce delay
                self.last_key_state = current_state
                active = self.toggled_active

            if active:
                frame = self.camera.get_latest_frame()
                if frame is not None and detect_color(frame, self.cmin, self.cmax):
                    current_time = time.time()
                    if current_time - self.last_shot_time >= self.shooting_rate:
                        self.queue.put("Shoot")
                        self.last_shot_time = current_time
                time.sleep(1 / self.fps)
            else:
                time.sleep(0.01)

class TriggerBotThread(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    def __init__(self, settings, fov, shooting_rate, fps, hsv_range, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.fov = fov
        self.shooting_rate = shooting_rate
        self.fps = fps
        self.hsv_range = hsv_range
        self.triggerbot = None
        self.shoot_queue = Queue()
        self.shoot_process = None

    def run(self):
        self.shoot_process = Process(target=simulate_shoot, args=(self.shoot_queue, self.settings))
        self.shoot_process.start()
        self.triggerbot = Triggerbot(self.shoot_queue, self.settings, self.fov, self.hsv_range, self.shooting_rate, self.fps)
        self.triggerbot.run()

    def stop(self):
        shutdown_event.set()
        if self.shoot_process:
            self.shoot_process.terminate()
            self.shoot_process.join()
        if self.triggerbot and hasattr(self.triggerbot, 'camera'):
            self.triggerbot.camera.stop()
        self.terminate()

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
        self.setFixedSize(620, 720)
        self.scanning_in_progress = False
        self.last_f5 = 0
        self.trigger_bot_thread = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        mainLayout = QtWidgets.QVBoxLayout(central_widget)
        mainLayout.setContentsMargins(20, 20, 20, 20)
        mainLayout.setSpacing(20)

        # Header with gradient background
        header = QtWidgets.QFrame()
        header.setFixedHeight(100)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #121212, stop:1 #32CD32);
                border-radius: 10px;
            }
        """)
        headerLayout = QtWidgets.QHBoxLayout(header)
        headerLabel = QtWidgets.QLabel("GamerFun Valo Menu V2")
        headerLabel.setStyleSheet("color: white; font-size: 32px; font-weight: bold;")
        headerLabel.setAlignment(QtCore.Qt.AlignCenter)
        headerLayout.addWidget(headerLabel)
        mainLayout.addWidget(header)

        # Agent Selection Group
        agentGroup = QtWidgets.QGroupBox("Agent Selection")
        agentLayout = QtWidgets.QVBoxLayout(agentGroup)   # Changed to QVBoxLayout

        # First row: Label and ComboBox
        agentSelectionLayout = QtWidgets.QHBoxLayout()
        agentLabel = QtWidgets.QLabel("Select Agent:")
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
        agentSelectionLayout.addWidget(agentLabel)
        agentSelectionLayout.addWidget(self.agent_combo)

        # Add the first row layout to the main agent group layout
        agentLayout.addLayout(agentSelectionLayout)

        # Second row: Informational Label
        agentInfoLabel = QtWidgets.QLabel("Press F5 before you enter the agent picking phase")
        agentInfoLabel.setStyleSheet("color: #32CD32; font-style: italic;")
        agentInfoLabel.setAlignment(QtCore.Qt.AlignCenter)
        agentLayout.addWidget(agentInfoLabel)

        mainLayout.addWidget(agentGroup)

        # TriggerBot Settings Group
        triggerGroup = QtWidgets.QGroupBox("TriggerBot Settings")
        triggerLayout = QtWidgets.QHBoxLayout(triggerGroup)
        self.triggerbot_checkbox = QtWidgets.QCheckBox("Enable TriggerBot")
        triggerLayout.addWidget(self.triggerbot_checkbox)
        keyLabel = QtWidgets.QLabel("Activation Key:")
        self.trigger_key_combo = QtWidgets.QComboBox()
        self.trigger_key_combo.addItems(list(self.key_map.keys()))
        inv_key_map = {v: k for k, v in self.key_map.items()}
        initial_key = inv_key_map.get(self.config.get("keybind", 164), "Alt")
        self.trigger_key_combo.setCurrentText(initial_key)
        triggerLayout.addWidget(keyLabel)
        triggerLayout.addWidget(self.trigger_key_combo)
        modeLabel = QtWidgets.QLabel("Mode:")
        self.hold_radio = QtWidgets.QRadioButton("Hold")
        self.toggle_radio = QtWidgets.QRadioButton("Toggle")
        if self.config.get("trigger_mode", "hold") == "toggle":
            self.toggle_radio.setChecked(True)
        else:
            self.hold_radio.setChecked(True)
        modeLayout = QtWidgets.QHBoxLayout()
        modeLayout.addWidget(modeLabel)
        modeLayout.addWidget(self.hold_radio)
        modeLayout.addWidget(self.toggle_radio)
        triggerLayout.addLayout(modeLayout)
        mainLayout.addWidget(triggerGroup)

        # Performance Settings Group
        perfGroup = QtWidgets.QGroupBox("Performance Settings")
        perfLayout = QtWidgets.QFormLayout(perfGroup)
        self.fov_spin = QtWidgets.QDoubleSpinBox()
        self.fov_spin.setRange(1.0, 50.0)
        self.fov_spin.setValue(self.config.get("fov", 5.0))
        self.fov_spin.setSingleStep(0.5)
        self.fov_spin.setDecimals(1)  # e.g. 5.0
        perfLayout.addRow("FOV:", self.fov_spin)
        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(10.0, 500.0)
        self.delay_spin.setValue(self.config.get("shooting_rate", 65.0))
        self.delay_spin.setSingleStep(5.0)
        self.delay_spin.setDecimals(0)  # e.g. 65
        perfLayout.addRow("Delay (ms):", self.delay_spin)
        # New toggle to enable/disable random press duration
        self.enable_press_duration_checkbox = QtWidgets.QCheckBox("Enable Random Press Duration")
        self.enable_press_duration_checkbox.setChecked(self.config.get("enable_press_duration", True))
        perfLayout.addRow(self.enable_press_duration_checkbox)
        self.press_min_spin = QtWidgets.QDoubleSpinBox()
        self.press_min_spin.setRange(0.01, 1.00)
        self.press_min_spin.setValue(self.config.get("press_duration_min", 0.01))
        self.press_min_spin.setSingleStep(0.01)
        self.press_min_spin.setDecimals(2)
        perfLayout.addRow("Press Duration Min (s):", self.press_min_spin)
        self.press_max_spin = QtWidgets.QDoubleSpinBox()
        self.press_max_spin.setRange(0.01, 1.00)
        self.press_max_spin.setValue(self.config.get("press_duration_max", 0.05))
        self.press_max_spin.setSingleStep(0.01)
        self.press_max_spin.setDecimals(2)
        perfLayout.addRow("Press Duration Max (s):", self.press_max_spin)
        self.block_movement_checkbox = QtWidgets.QCheckBox("Block Movements While Shooting")
        self.block_movement_checkbox.setChecked(self.config.get("block_movements", False))
        perfLayout.addRow(self.block_movement_checkbox)
        mainLayout.addWidget(perfGroup)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        mainLayout.addWidget(self.log_text)

        # Connect signals
        self.triggerbot_checkbox.stateChanged.connect(self.toggle_triggerbot)
        self.trigger_key_combo.currentIndexChanged.connect(self.update_keybind)
        self.fov_spin.valueChanged.connect(self.update_config)
        self.delay_spin.valueChanged.connect(self.update_config)
        self.press_min_spin.valueChanged.connect(self.update_config)
        self.press_max_spin.valueChanged.connect(self.update_config)
        self.block_movement_checkbox.stateChanged.connect(self.update_config)
        self.enable_press_duration_checkbox.stateChanged.connect(self.on_press_duration_toggle)
        self.hold_radio.toggled.connect(self.update_trigger_mode)
        self.toggle_radio.toggled.connect(self.update_trigger_mode)

        self.hotkey_timer = QtCore.QTimer(self)
        self.hotkey_timer.timeout.connect(self.check_hotkey)
        self.hotkey_timer.start(50)

        self.apply_styles()
        # Set the enabled state of press duration inputs based on the checkbox
        self.on_press_duration_toggle(self.enable_press_duration_checkbox.checkState())

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
        self.press_min_spin.setEnabled(enabled)
        self.press_max_spin.setEnabled(enabled)
        self.config["enable_press_duration"] = enabled
        save_config(self.config)

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
        self.config["fov"] = self.fov_spin.value()
        self.config["shooting_rate"] = self.delay_spin.value()
        self.config["press_duration_min"] = self.press_min_spin.value()
        self.config["press_duration_max"] = self.press_max_spin.value()
        self.config["block_movements"] = self.block_movement_checkbox.isChecked()
        save_config(self.config)

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.append(f"<span style='color:#32CD32'>{timestamp}</span>{message}")

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
        self.scanning_in_progress = True
        self.worker = ScanningWorker(agent_name, agent_uuid)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.session_finished)
        self.worker.start()

    def session_finished(self):
        self.scanning_in_progress = False
        self.log("Instalock process completed")

    def toggle_triggerbot(self, state):
        if state == QtCore.Qt.Checked:
            self.update_keybind()
            self.config["fov"] = self.fov_spin.value()
            self.config["shooting_rate"] = self.delay_spin.value()
            save_config(self.config)
            self.trigger_bot_thread = TriggerBotThread(
                settings=self.config,
                fov=self.fov_spin.value(),
                shooting_rate=self.delay_spin.value(),
                fps=self.config.get("fps", 200.0),
                hsv_range=self.config.get("hsv_range", [[30,125,150],[30,255,255]])
            )
            self.trigger_bot_thread.log_signal.connect(self.log)
            self.trigger_bot_thread.start()
            self.log("TriggerBot ACTIVATED")
        else:
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
# UI-DO-NOT-OBFUSCATE-END

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
