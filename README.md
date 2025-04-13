# Valorant Trigger Bot & Spoofer

**Disclaimer:**  
This project is for educational and research purposes only. Using this software in online games may violate the game's terms of service and can result in account bans or other penalties. Use at your own risk. The author is not responsible for any misuse or damage resulting from the use of this software.

## Overview

Valorant Trigger Bot & Spoofer is designed to assist players by:
- **Auto Trigger:** Automatically firing when an enemy is on your crosshair.
- **Insta Lock:** Instantly locking onto the target during the operator picking phase.
- **Advanced Code Spoofing:** Using AST-based transformation to completely randomize function names, variable names, and code structure on every run.
- **GPU Acceleration:** Utilizing CUDA and PyTorch for dramatically improved detection performance on compatible systems.

The tool features a user-friendly interface (built with PyQt5) that allows you to adjust settings such as field of view (FOV), shooting delay, and other performance parameters.

- **Official Website:** [GamerFun Valorant Trigger Bot, Instant Lock and Spoofer
](https://www.gamerfun.club/gamerfun-valorant-trigger-bot-instant-lock-and-spoofer)
- **UnknownCheats Forum:** [[Release] ⭐GamerFun Valo: Valorant Trigger Bot, Instant Lock and Spoofer](https://www.unknowncheats.me/forum/valorant/690063-gamerfun-valo-valorant-trigger-bot-instant-lock-spoofer.html)
- **GitHub Repository:** [valorant-instalock-triggerbot-spoofer](https://github.com/WANASX/valorant-instalock-triggerbot-spoofer)

## Features

- **Auto Trigger with Dynamic Delay:**  
  When an enemy is detected on your crosshair, the bot simulates a left mouse click with configurable random delay between min/max values for more human-like behavior.
  
- **GPU-Accelerated Detection:**  
  Utilizes CUDA and PyTorch for high-performance color detection, with automatic fallback to CPU if needed.
  
- **Insta Lock:**  
  Quickly locks onto a target during the operator selection phase, with clear warning about potential ban risks.
  
- **Advanced Code Spoofing:**  
  Uses Abstract Syntax Tree (AST) transformation to completely randomize function and variable names, adds random spacing, comments, and no-op code to make detection nearly impossible.
  
- **Smart Acceleration:**  
  Automatically benchmarks and selects the best processing method (GPU or CPU) based on your hardware.
  
- **Improved Interface:**  
  Enhanced GUI with thematic options, better settings organization, and clear performance indicators.

## New in Version 3.0

- **AST-based Code Transformation:** Complete rewrite of the spoofer to use Python's AST for robust code transformation.
- **GPU Acceleration:** Added CUDA and PyTorch support for dramatically improved detection performance.
- **Smart Benchmarking:** Automatically tests and selects the best acceleration method for your system.
- **Randomized Shooting Delay:** Configure min/max shooting delay values for more human-like behavior.
- **Block Movement Option:** Option to automatically release movement keys (WASD) while shooting for better accuracy.
- **Enhanced Safety:** Improved error handling, configuration validation, and compatibility checks.
- **Better UI:** Redesigned interface with theme options and better visualization of performance.

## Requirements

- **Operating System:** Windows  
- **Python Version:** 3.9+  
- **Dependencies:**  
  - PyQt5  
  - opencv-python (with CUDA support recommended)
  - numpy  
  - pytorch (with CUDA support recommended)
  - pyautogui  
  - pypiwin32  
  - bettercam  
  - astor (for code transformation)

## Installation

1. **Clone the repository:**
```
git clone https://github.com/WANASX/valorant-instalock-triggerbot-spoofer.git
cd valorant-instalock-triggerbot-spoofer
```
2. **Install dependencies:**
```
pip install opencv-python numpy pyautogui pywin32 PyQt5 bettercam torch astor
```
## Usage

1. **Run as Administrator:**  
   Run the Main.bat file with administrator privileges for proper mouse event simulation.

2. **Configure Settings:**  
   Launch the application using Main.bat. Use the GUI to adjust settings such as FOV, min/max shooting delay, and GPU acceleration options.

3. **Start the TriggerBot:**  
   The TriggerBot is enabled by default. Use the configured hotkey (Alt by default) to activate triggering when enemies are detected.

4. **InstaLock Agent:**  
   Press F5 before entering agent selection to automatically lock your chosen agent.
   
5. **Code Spoofing:**  
   Every time you run the application, it generates a uniquely randomized version with transformed function names, variable names, and code structure.
   
6. **Enemy Highlight Color:**
   For best detection, go to Valorant settings and change enemy highlight color to Yellow (protanopia).

## File Structure

- **main.bat:** Launcher script that runs the application with administrator privileges.
- **menu.py:** Contains the core UI and trigger bot logic.
- **main.py:** Advanced spoofer that transforms menu.py using AST and generates a unique version.
- **temp/**: Directory where obfuscated files are generated.
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
