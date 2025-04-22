import os
import sys
import time
import json
import random
import subprocess
import uuid
import hashlib
import base64
import hmac
from hashlib import sha256
from cryptography.fernet import Fernet
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt
import pyautogui
import pyperclip

# --- CONFIG ---
TOKEN_FILE = ".ad_key"
TOKEN_DURATION = 12 * 3600  # 12 hours
DEFAULT_SECRET = b"supersecretkey1234567890"
MARKER_FILE = os.path.join(os.environ.get("APPDATA", "."), ".ad_key_marker")
LAST_TIME_FILE = os.path.join(os.environ.get("APPDATA", "."), ".ad_key_lasttime")
SITE_VISITED_FILE = ".site_visited"
AD_CLICKED_FILE = ".ad_clicked"
PAUSE_ADS_FILE = ".pause_ads"
PAUSE_DURATION = 40
DOMAIN = "gamerfun.club"
ADSENSE_PATTERNS = ["doubleclick.net", "ads.google.com", "/adclick"]
KEYWORDS_FILE = "Website_Keywords.json"

# --- UTILS ---
def get_machine_guid():
    try:
        output = subprocess.check_output('wmic csproduct get uuid', shell=True).decode()
        lines = output.splitlines()
        for line in lines:
            if line.strip() and line.strip().lower() != 'uuid':
                return line.strip()
    except Exception:
        return "noguid"
    return "noguid"

def get_username():
    return os.environ.get("USERNAME", "nouser")

def get_mac_hash():
    try:
        mac = uuid.getnode()
        return hashlib.sha256(str(mac).encode()).hexdigest()
    except Exception:
        return "nomac"

def get_secret():
    val = os.environ.get("AD_CLICK_SECRET")
    if val is not None:
        return val.encode() if isinstance(val, str) else val
    return DEFAULT_SECRET

def generate_token():
    issued_at = int(time.time())
    guid = get_machine_guid()
    user = get_username()
    mac = get_mac_hash()
    payload = f"{issued_at}:{guid}:{user}:{mac}".encode()
    h = hmac.new(get_secret(), payload, sha256).digest()
    token_data = base64.urlsafe_b64encode(payload + b":" + h)
    fernet_key = base64.urlsafe_b64encode(sha256(get_secret()).digest())[:32]
    f = Fernet(base64.urlsafe_b64encode(fernet_key))
    return f.encrypt(token_data).decode()

def store_token(token):
    try:
        with open(TOKEN_FILE, "w") as f:
            f.write(token)
        with open(MARKER_FILE, "w") as f:
            f.write(str(int(time.time())))
        with open(LAST_TIME_FILE, "w") as f:
            f.write(str(int(time.time())))
        print(f"[DEBUG] Token and marker files written: {TOKEN_FILE}, {MARKER_FILE}, {LAST_TIME_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to write token/marker files: {e}")

def load_token():
    try:
        if not os.path.exists(TOKEN_FILE):
            print(f"[DEBUG] {TOKEN_FILE} does not exist.")
            return ""
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
        print(f"[DEBUG] Loaded token from {TOKEN_FILE}")
        return token
    except Exception as e:
        print(f"[ERROR] Failed to load token: {e}")
        return ""

def delete_token():
    for f in [TOKEN_FILE, MARKER_FILE, LAST_TIME_FILE]:
        if os.path.exists(f):
            os.remove(f)

def get_token_issued_at(token):
    try:
        fernet_key = base64.urlsafe_b64encode(sha256(get_secret()).digest())[:32]
        f = Fernet(base64.urlsafe_b64encode(fernet_key))
        token_data = f.decrypt(token.encode(), ttl=TOKEN_DURATION)
        payload, h = base64.urlsafe_b64decode(token_data).rsplit(b":", 1)
        parts = payload.decode().split(":")
        issued_at = int(parts[0])
        guid = parts[1]
        user = parts[2]
        mac = parts[3]
        expected_h = hmac.new(get_secret(), payload, sha256).digest()
        if h != expected_h:
            return 0
        if guid != get_machine_guid() or user != get_username() or mac != get_mac_hash():
            return 0
        return issued_at
    except Exception:
        return 0

def get_token_time_left(token):
    issued_at = get_token_issued_at(token)
    if not issued_at:
        return 0
    left = TOKEN_DURATION - (int(time.time()) - issued_at)
    return max(0, left)

def is_token_valid(token):
    now = int(time.time())
    if os.path.exists(LAST_TIME_FILE):
        try:
            with open(LAST_TIME_FILE, "r") as f:
                last_seen = int(f.read().strip())
            if now < last_seen:
                print("[ERROR] System clock was set backwards. Please correct your system time.")
                return False
        except Exception as e:
            print(f"[ERROR] Reading LAST_TIME_FILE: {e}")
            return False
    try:
        with open(LAST_TIME_FILE, "w") as f:
            f.write(str(now))
    except Exception as e:
        print(f"[ERROR] Writing LAST_TIME_FILE: {e}")
        return False
    if not get_token_time_left(token) > 0:
        print("[DEBUG] Token expired or invalid time left.")
        return False
    if not os.path.exists(MARKER_FILE):
        print(f"[DEBUG] Marker file {MARKER_FILE} missing.")
        return False
    # Validate token matches this machine/user/mac
    issued_at = get_token_issued_at(token)
    if not issued_at:
        print("[DEBUG] Token machine/user/mac mismatch or HMAC invalid.")
        return False
    return True

def format_time_left(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}h {m}m {s}s"

# --- OVERLAY ---
class Overlay:
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.label = QLabel()
        self.label.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.label.setStyleSheet("background-color: #222; color: #fff; font-size: 18px; padding: 16px; border-radius: 8px; border: 4px solid #39ff14;")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.adjustSize()
        screen = self.app.primaryScreen().geometry()
        self.label.move(30, screen.height() - self.label.height() - 60)
        self.label.show()
        self.app.processEvents()
    def set(self, msg):
        self.label.setText(msg)
        self.label.adjustSize()
        self.app.processEvents()
    def close(self):
        self.label.close()
        self.app.processEvents()

# --- MITMPROXY ADDON ---
if "mitmproxy" in sys.modules or any("mitmproxy" in arg for arg in sys.argv):
    from mitmproxy import http
    class AdClickAddon:
        def __init__(self):
            self.site_visited = False
            self.ad_clicked = False
        def request(self, flow: http.HTTPFlow):
            url = flow.request.pretty_url.lower()
            host = flow.request.host.lower()
            if (not os.path.exists(SITE_VISITED_FILE)) and (DOMAIN in host):
                with open(SITE_VISITED_FILE, "w") as f:
                    f.write("1")
                with open(PAUSE_ADS_FILE, "w") as f:
                    f.write(str(time.time()))
            if (not os.path.exists(AD_CLICKED_FILE)) and (not is_paused()) and any(p in url for p in ADSENSE_PATTERNS):
                with open(AD_CLICKED_FILE, "w") as f:
                    f.write("1")
    addons = [AdClickAddon()]
    def is_paused():
        if not os.path.exists(PAUSE_ADS_FILE):
            return False
        try:
            with open(PAUSE_ADS_FILE, "r") as f:
                t = float(f.read().strip())
            return (time.time() - t) < PAUSE_DURATION
        except Exception:
            return False
else:
    def is_paused():
        if not os.path.exists(PAUSE_ADS_FILE):
            return False
        try:
            with open(PAUSE_ADS_FILE, "r") as f:
                t = float(f.read().strip())
            return (time.time() - t) < PAUSE_DURATION
        except Exception:
            return False

def site_visited():
    return os.path.exists(SITE_VISITED_FILE)

def ad_clicked():
    return os.path.exists(AD_CLICKED_FILE)

def reset_flags():
    for f in [SITE_VISITED_FILE, AD_CLICKED_FILE, PAUSE_ADS_FILE]:
        if os.path.exists(f):
            os.remove(f)

def cleanup():
    for f in [SITE_VISITED_FILE, AD_CLICKED_FILE, PAUSE_ADS_FILE, TOKEN_FILE, MARKER_FILE, LAST_TIME_FILE]:
        if os.path.exists(f):
            os.remove(f)

# --- CHROME CONTROL ---
def kill_chrome():
    if sys.platform == "win32":
        subprocess.call("taskkill /F /IM chrome.exe", shell=True)
    else:
        subprocess.call(["pkill", "chrome"])

def find_chrome_path():
    possible_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def launch_chrome_and_search(query):
    chrome_path = find_chrome_path()
    if not chrome_path:
        print("[ERROR] Chrome not found. Please install Google Chrome.")
        sys.exit(1)
    subprocess.Popen([
        chrome_path,
        "--proxy-server=127.0.0.1:8080",
        "https://google.com"
    ])
    time.sleep(4)
    pyautogui.typewrite(query, interval=0.11)  # Human-like typing speed
    pyautogui.press('enter')

# --- MAIN LOGIC ---
def main():
    # Step 1: Token check (before anything else)
    token = load_token()
    if token and is_token_valid(token):
        overlay = Overlay()
        left = get_token_time_left(token)
        msg = f"[INFO] Valid key found. Loader active for {format_time_left(left)}."
        overlay.set(msg)
        print(msg)
        time.sleep(4)
        overlay.close()
        # Run main.py after successful key check
        subprocess.Popen([sys.executable, "main.py"])
        sys.exit(0)
    else:
        delete_token()

    # Step 2: Show overlay and kill Chrome
    overlay = Overlay()
    for i in range(5, 0, -1):
        overlay.set(f"Closing Chrome in {i}...")
        time.sleep(1)
    kill_chrome()
    overlay.set("Starting loader...")
    time.sleep(1)
    reset_flags()

    # Step 3: Start mitmproxy as a subprocess
    script_path = os.path.abspath(__file__)
    mitm_proc = subprocess.Popen([
        "mitmdump", "-s", script_path, "--listen-port", "8080", "--set", "console_eventlog_verbosity=error"
    ])
    time.sleep(2)

    # Step 4: Pick random keyword
    with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
        keywords = json.load(f)
    search_word = random.choice(keywords)
    search_query = f"{search_word} site:{DOMAIN}"
    overlay.set(f"Searching: {search_query}")
    time.sleep(1)

    # Step 5: Launch Chrome and search
    launch_chrome_and_search(search_query)
    overlay.set("Waiting for Google to load and search...")
    time.sleep(1)
    # Step 6: 7s scroll timer
    for i in range(7, 0, -1):
        overlay.set(f"Please wait and scroll for {i} seconds...")
        time.sleep(1)
    overlay.set("Please click on any search result for your site to continue...")
    # Step 7: Wait for site visit
    while not site_visited():
        time.sleep(0.5)
    # Step 8: 40s ad timer
    for i in range(40, 0, -1):
        overlay.set(f"Scroll and wait... {i} seconds left before clicking any ad.")
        time.sleep(1)
    overlay.set("You may now click any ad on the site.")
    # Step 9: Wait for ad click (do not skip this loop)
    while not ad_clicked():
        overlay.set("Waiting for you to click an ad...")
        time.sleep(0.5)
    overlay.set("Ad click detected! Please wait 15 seconds...")
    for i in range(15, 0, -1):
        overlay.set(f"Please wait... Saving your key in {i} seconds.")
        time.sleep(1)
    token = generate_token()
    store_token(token)
    overlay.set("Mission complete! Your key is saved.")
    time.sleep(3)
    overlay.close()
    # Step 11: Terminate mitmproxy
    mitm_proc.terminate()
    try:
        mitm_proc.wait(timeout=5)
    except Exception:
        mitm_proc.kill()
    # After getting new key, run main.py
    subprocess.Popen([sys.executable, "main.py"])
    sys.exit(0)

if __name__ == "__main__":
    # Only run main if not being run by mitmproxy
    if not ("mitmproxy" in sys.modules or any("mitmproxy" in arg for arg in sys.argv)):
        main() 