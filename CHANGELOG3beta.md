# CHANGELOG

## Version 3.0.0 (2023-07-15)

### Major Features

#### Advanced Spoofer System
- **AST-Based Code Transformation**: Complete rewrite of spoofer using Python's Abstract Syntax Tree (AST) for deep code transformation
- **Function & Variable Renaming**: Randomizes function and variable names throughout the codebase
- **Code Layout Randomization**: Varies spacing, line breaks, and code structure on each run
- **Dead Code Insertion**: Adds harmless no-op statements and random constants
- **Syntax Validation**: Ensures transformed code remains syntactically valid through multi-stage verification

#### GPU Acceleration System
- **CUDA & PyTorch Support**: Added GPU-accelerated color detection for dramatically improved performance
- **Smart Acceleration**: Automatically benchmarks and selects the best processing method (GPU vs CPU)
- **Fallback Mechanism**: Gracefully falls back to CPU if GPU processing fails
- **Performance Monitoring**: Real-time FPS tracking and diagnostics

#### Enhanced TriggerBot
- **Random Shooting Delay**: Configure min/max shooting delay for more human-like behavior
- **Movement Blocking**: Option to automatically release movement keys (WASD) while shooting
- **Auto-Start**: TriggerBot is enabled by default on startup for convenience
- **Random Press Duration**: Configurable random duration for mouse clicks

### UI Improvements
- **Theme Support**: Added multiple theme options (Dark, Light, Custom Green)
- **GPU Status Display**: Clear visualization of GPU availability and status
- **Benchmark Tool**: Built-in performance testing for different acceleration methods
- **Enhanced Settings Organization**: Better categorization of configuration options
- **Ban Risk Warning**: Clear warning about risks of using agent instalock feature

### Technical Improvements
- **Error Handling**: Improved exception catching and error reporting
- **Code Safety**: Prevented critical function names from being randomized
- **Memory Management**: Better cleanup of GPU resources
- **Configuration System**: Expanded configuration options with validation
- **Compatibility Checks**: Automatic detection of available GPU support

### Config Changes
- Added `min_shooting_rate` and `max_shooting_rate` for randomized delays
- Added `use_gpu` for GPU acceleration toggle
- Added `auto_fallback_to_cpu` for graceful fallback
- Added `smart_acceleration` for automatic performance optimization
- Added `test_mode` for benchmark comparison
- Added `theme` setting for UI customization

### Bug Fixes
- Fixed delay discrepancies in triggerbot activation
- Fixed potential memory leaks in camera handling
- Fixed UI hanging issues when changing settings
- Fixed error handling in multi-process communication

### Code Structure
- Reorganized codebase for better maintainability
- Implemented threading safety improvements
- Added comprehensive comments and docstrings
- Improved module organization and code reuse

### Removed Features
- Removed dependency on agent images folder
- Simplified some advanced options for better usability

## Compatibility Notes
- Now requires Python 3.9+ (up from 3.7+)
- Requires additional dependencies: PyTorch and astor
- CUDA support is optional but recommended for best performance
