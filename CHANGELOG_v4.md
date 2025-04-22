# CHANGELOG_v4.md

## Version 4.0.0

### Major New Features

#### AimLock Module
- **Brand new AimLock system** for advanced color-based aim lock.
- Supports both **toggle** and **hold** activation modes.
- **Adaptive scan** and **spiral scan** patterns for improved target acquisition.
- **Debug mode** for visualizing detection and troubleshooting.
- **GPU/CPU support** with automatic switching and benchmarking.
- **Sticky aim** using mouse movement hooks for more natural tracking.
- **Highly configurable**: scan area, strength, color, tolerance, refresh rate, and more.
- **Seamless integration** with the main UI and config system.

#### Profile System
- **Completely new profile management system**:
  - Create, edit, import, and export profiles for different weapons or playstyles.
  - **Auto profile detection**: Automatically switches profiles based on detected weapon using OCR.
  - **Profile migration**: Old profiles are automatically updated to the new schema.
  - **Shortcut mapping**: Assign hotkeys to quickly switch between profiles.
  - **Profile editor**: Full-featured UI for managing all profile settings.
  - **Profiles stored as JSON** in the `profiles/` directory for easy sharing and backup.

#### UI/UX Improvements
- **Modernized interface** with new tabs for AimLock, Profile Management, and Debug Logs.
- **Theme support**: Dark, Light, and Custom Green themes.
- **Improved settings organization** and tooltips for all options.
- **Debug report generation**: Export a detailed report for troubleshooting.
- **Status bar** with real-time GPU/CPU status and performance info.
- **Resizable, more accessible window** with better default sizing.

#### Performance & Technical
- **Enhanced benchmarking**: Compare CPU vs GPU performance for detection.
- **More robust GPU/CPU switching** with clear feedback and fallback.
- **Improved error handling** and logging throughout the application.
- **Modularized codebase**: Core logic split into `aimlock.py`, `triggerbot.py`, `shared_scanner.py`, and more.
- **Config migration**: Automatic update of old configs and profiles to new keys and formats.

#### TriggerBot & Instalock
- All v3 features retained and improved.
- **TriggerBot**: More robust, with better FOV handling, random delay, and movement blocking.
- **Instalock**: Improved region detection and agent locking logic.

### Configuration Changes
- **New config keys** for AimLock, profile system, and advanced settings.
- **Profiles** now support all AimLock and TriggerBot options per weapon.
- **Backward compatibility**: Old configs and profiles are automatically migrated.

### Bug Fixes & Quality of Life
- Fixed various UI hangs, memory leaks, and error handling issues.
- Improved multi-threading and resource cleanup.
- More comprehensive comments and docstrings for maintainability.

### File Structure Changes
- **New files**: `aimlock.py`, `triggerbot.py`, `shared_scanner.py`, `profiles/`
- **Profiles**: Each weapon/profile is a separate JSON file in `profiles/`.
- **New UI image**: `Valo_Menu_V4.PNG`

---

## Upgrade Notes

- **Python 3.9+ is now required.**
- **PyTorch and astor** are required dependencies.
- **CUDA support** is optional but recommended for best performance.
- **Old profiles/configs** will be migrated automatically on first run.

---

## Summary

Version 4.0 is a major upgrade, introducing a powerful AimLock system, a flexible profile manager, a modernized UI, and significant performance and reliability improvements. All previous features are retained and enhanced, making this the most advanced and user-friendly release yet. 