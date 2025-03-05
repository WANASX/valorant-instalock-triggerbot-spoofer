# Valorant Trigger Bot & Spoofer

**Disclaimer:**  
This project is for educational and research purposes only. Using this software in online games may violate the game’s terms of service and can result in account bans or other penalties. Use at your own risk. The author is not responsible for any misuse or damage resulting from the use of this software.

## Overview

Valorant Trigger Bot & Spoofer is designed to assist players by:
- **Auto Trigger:** Automatically firing when an enemy is on your crosshair.
- **Insta Lock:** Instantly locking onto the target during the operator picking phase.
- **Code Spoofing:** Obfuscating and randomizing the internal code on every run to make each execution unique.

The tool features a user-friendly interface (built with PyQt5) that allows you to adjust settings such as field of view (FOV) and shooting delay.

- **Official Website:** [GamerFun Valorant Trigger Bot, Instant Lock and Spoofer
](https://www.gamerfun.club/ai-aimbot-triggerbot-shooter-games)
- **UnknownCheats Forum:** [[Release] GamerFun AI Menu: All mouse Aimbot, Triggerbot & Recoil Control Using LGUB Drivers](https://www.unknowncheats.me/forum/rainbow-six-siege/671029-gamerfun-ai-menu-mouse-aimbot-triggerbot-recoil-control-using-lgub-drivers.html)
- **GitHub Repository:** [valorant-instalock-triggerbot-spoofer](https://github.com/WANASX/valorant-instalock-triggerbot-spoofer)

## Features

- **Auto Trigger:**  
  When an enemy is detected on your crosshair, the bot simulates a left mouse click.
  
- **Insta Lock:**  
  Quickly locks onto a target during the operator selection phase.
  
- **Unique Code Spoofing:**  
  On every run, the tool obfuscates its internal logic and generates a unique file to help bypass static signature detection.
  
- **GUI Configuration:**  
  Easily configure settings like FOV and shooting delay via the graphical user interface.

## Requirements

- **Operating System:** Windows  
- **Python Version:** 3.9+  
- **Dependencies:**  
  - PyQt5  
  - opencv-python  
  - numpy  
  - pyautogui  
  - pypiwin32  
  - bettercam  

## Installation

1. **Clone the repository:**
```
git clone https://github.com/WANASX/valorant-instalock-triggerbot-spoofer.git
cd valorant-instalock-triggerbot-spoofer
```
2. **Install dependencies:**
```
pip install opencv-python numpy pyautogui pywin32 PyQt5 bettercam
```
## Usage

1. **Run as Administrator:**  
   Run the The Main.bat requires administrator privileges for proper mouse event simulation.

2. **Configure Settings:**  
   Launch the Main.bat. Use the GUI to adjust settings such as FOV and shooting delay.

3. **Start the Bot:**  
   Press F5 to initiate scanning. When an enemy is detected on your crosshair, the bot fires automatically.

4. **Code Spoofing:**  
   Each run, the spoofer obfuscates the internal code (except for the UI design) and generates a uniquely named file in the `temp/` folder.
   
6. **Enemy Highlight Color**
   Go to Valorant settings and change enemy highlight color to Yellow (protanopia)

## File Structure

- **main.bat:** Contains the launcher to make it easy to laucnh.
- **menu.py:** Contains the original UI and trigger bot logic.
- **main.py:** The spoofer/randomizer that obfuscates `menu.py` and launches the unique version.
- **temp/**: Directory where obfuscated files are generated.
- **Agents/**: Directory containing agent images for scanning.
- **config.json:** Configuration file storing user settings.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for enhancements or bug fixes.

---

## Support & Donations

If you would like to support this project, you can donate **USDT Tron (TRC20)** to:
```
TDiVQzShforoR5XgWXfKuPhPhdgPypXAgB
```

For questions or support, contact **support@gamerfun.club**.



**Note:** This project is intended for research and educational purposes only. Use responsibly and at your own risk.
