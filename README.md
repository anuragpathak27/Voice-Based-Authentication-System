# Voice Authentication System

A secure voice-based authentication system that uses voice biometrics to identify and authenticate users. This application provides both enrollment and recognition capabilities with a user-friendly GUI.

## Features

- **Voice Enrollment**: Register users by recording their voice samples
- **Voice Recognition**: Authenticate users by matching their voice patterns
- **Spectrogram Visualization**: View voice sample spectrograms for analysis
- **User Management**: View and manage enrolled users
- **Interactive Console**: System feedback and logging

## Technologies Used

- Python 3.x
- Tkinter (GUI)
- Matplotlib (Spectrogram visualization)
- Sounddevice/Soundfile (Audio recording/playback)
- NumPy (Audio processing)
- SciPy (Signal processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anuragpathak27/voice-authentication-system.git
   cd voice-authentication-system

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the application:
   ```bash
   python main.py
