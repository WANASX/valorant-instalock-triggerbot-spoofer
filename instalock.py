import time
import threading
from PyQt5 import QtCore
from concurrent.futures import ThreadPoolExecutor, as_completed
import win32api
import win32con

# ------------------- Agent Locking (Instalock) Logic -------------------
def initialize_client(region):
    try:
        from valclient import Client  # Import directly from valclient
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

# Agent dictionary for reference and use in menu.py
AGENT_DICT = {
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