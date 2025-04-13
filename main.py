import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from voice_auth import enroll, recognize
import parameters as p
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from preprocess import get_fft_spectrum
from feature_extraction import buckets
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime

class VoiceAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Authentication System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f5")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background="#f0f2f5")
        self.style.configure('TLabel', background="#f0f2f5", font=("Arial", 12))
        self.style.configure('TButton', font=("Arial", 12), padding=5)
        self.style.map('TButton', 
                      foreground=[('active', 'white'), ('!active', 'white')],
                      background=[('active', '#3498db'), ('!active', '#2980b9')])
        
        # Create container frame
        self.container = tk.Frame(root, bg="#f0f2f5")
        self.container.pack(fill="both", expand=True)
        
        # Initialize pages
        self.pages = {}
        for Page in (VoiceLoginPage, VoiceSignupPage, MainPage):
            page_name = Page.__name__
            page = Page(parent=self.container, controller=self)
            self.pages[page_name] = page
            page.grid(row=0, column=0, sticky="nsew")
        
        self.show_page("VoiceLoginPage")
    
    def show_page(self, page_name):
        page = self.pages[page_name]
        page.tkraise()
        if page_name == "MainPage":
            page.refresh_enrolled_users()

class VoiceLoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#f0f2f5")
        self.controller = controller
        self.recording = False
        self.frames = []
        
        # Header
        tk.Label(self, text="Voice Authentication Login", font=("Arial", 24, "bold"), 
                bg="#f0f2f5", fg="#2c3e50").pack(pady=40)
        
        # Login Frame
        login_frame = ttk.Frame(self)
        login_frame.pack(pady=20)
        
        # Username
        ttk.Label(login_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.username_entry = ttk.Entry(login_frame, width=25)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Voice Login Button
        self.voice_login_btn = ttk.Button(login_frame, text="Record Voice", 
                                        command=self.toggle_recording)
        self.voice_login_btn.grid(row=1, columnspan=2, pady=10)
        
        # Spectrogram Display
        self.spectrogram_frame = tk.Frame(login_frame, bg="white", height=150)
        self.spectrogram_frame.grid(row=2, columnspan=2, pady=10, sticky="ew")
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Login", 
                  command=self.login).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Sign Up", 
                  command=lambda: controller.show_page("VoiceSignupPage")).pack(side="left", padx=10)
        
        # Status
        self.status_label = tk.Label(self, text="", bg="#f0f2f5", fg="#e74c3c")
        self.status_label.pack(pady=10)
        
        # Audio Config
        self.sample_rate = 16000
        self.duration = 3  # seconds
        self.audio_file = "temp_login.wav"
    
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.voice_login_btn.config(text="Recording... (3s)")
            self.frames = []
            
            # Start recording
            self.recording_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback
            )
            self.recording_stream.start()
            
            # Stop after duration
            self.root.after(self.duration * 1000, self.stop_recording)
        else:
            self.stop_recording()
    
    def audio_callback(self, indata, frames, time, status):
        self.frames.append(indata.copy())
    
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.voice_login_btn.config(text="Record Voice")
            self.recording_stream.stop()
            
            # Save recording
            audio_data = np.concatenate(self.frames)
            sf.write(self.audio_file, audio_data, self.sample_rate)
            
            # Show spectrogram
            self.show_spectrogram(self.audio_file)
            self.status_label.config(text="Voice sample recorded", fg="#2ecc71")
    
    def show_spectrogram(self, filepath):
        for widget in self.spectrogram_frame.winfo_children():
            widget.destroy()
        
        try:
            buckets_var = buckets(p.MAX_SEC, p.BUCKET_STEP, p.FRAME_STEP)
            spectrogram = get_fft_spectrum(filepath, buckets_var)
            
            fig = plt.Figure(figsize=(5, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            cax = ax.imshow(spectrogram, aspect='auto', origin='lower')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Spectrogram")
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=self.spectrogram_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            self.status_label.config(text=f"Spectrogram error: {str(e)}", fg="#e74c3c")
    
    def login(self):
        username = self.username_entry.get()
        
        if not username:
            self.status_label.config(text="Username required", fg="#e74c3c")
            return
        
        if not os.path.exists(self.audio_file):
            self.status_label.config(text="Please record voice sample", fg="#e74c3c")
            return
        
        try:
            # Check if user exists
            user_file = os.path.join(p.EMBED_LIST_FILE, f"{username}.npy")
            if not os.path.exists(user_file):
                self.status_label.config(text="User not found", fg="#e74c3c")
                return
            
            # Verify voice
            recognize(self.audio_file)
            self.controller.show_page("MainPage")
        except Exception as e:
            self.status_label.config(text=f"Login failed: {str(e)}", fg="#e74c3c")

class VoiceSignupPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#f0f2f5")
        self.controller = controller
        self.recording = False
        self.frames = []
        
        # Header
        tk.Label(self, text="Voice Authentication Signup", font=("Arial", 24, "bold"), 
                bg="#f0f2f5", fg="#2c3e50").pack(pady=40)
        
        # Signup Frame
        signup_frame = ttk.Frame(self)
        signup_frame.pack(pady=20)
        
        # Username
        ttk.Label(signup_frame, text="Username:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.username_entry = ttk.Entry(signup_frame, width=25)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Voice Record Button
        self.voice_record_btn = ttk.Button(signup_frame, text="Record Voice", 
                                         command=self.toggle_recording)
        self.voice_record_btn.grid(row=1, columnspan=2, pady=10)
        
        # Spectrogram Display
        self.spectrogram_frame = tk.Frame(signup_frame, bg="white", height=150)
        self.spectrogram_frame.grid(row=2, columnspan=2, pady=10, sticky="ew")
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Sign Up", 
                  command=self.signup).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Back to Login", 
                  command=lambda: controller.show_page("VoiceLoginPage")).pack(side="left", padx=10)
        
        # Status
        self.status_label = tk.Label(self, text="", bg="#f0f2f5", fg="#e74c3c")
        self.status_label.pack(pady=10)
        
        # Audio Config
        self.sample_rate = 16000
        self.duration = 5  # seconds for enrollment
        self.audio_file = "temp_signup.wav"
    
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.voice_record_btn.config(text="Recording... (5s)")
            self.frames = []
            
            # Start recording
            self.recording_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback
            )
            self.recording_stream.start()
            
            # Stop after duration
            self.root.after(self.duration * 1000, self.stop_recording)
        else:
            self.stop_recording()
    
    def audio_callback(self, indata, frames, time, status):
        self.frames.append(indata.copy())
    
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.voice_record_btn.config(text="Record Voice")
            self.recording_stream.stop()
            
            # Save recording
            audio_data = np.concatenate(self.frames)
            sf.write(self.audio_file, audio_data, self.sample_rate)
            
            # Show spectrogram
            self.show_spectrogram(self.audio_file)
            self.status_label.config(text="Voice sample recorded", fg="#2ecc71")
    
    def show_spectrogram(self, filepath):
        for widget in self.spectrogram_frame.winfo_children():
            widget.destroy()
        
        try:
            buckets_var = buckets(p.MAX_SEC, p.BUCKET_STEP, p.FRAME_STEP)
            spectrogram = get_fft_spectrum(filepath, buckets_var)
            
            fig = plt.Figure(figsize=(5, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            cax = ax.imshow(spectrogram, aspect='auto', origin='lower')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Spectrogram")
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=self.spectrogram_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            self.status_label.config(text=f"Spectrogram error: {str(e)}", fg="#e74c3c")
    
    def signup(self):
        username = self.username_entry.get()
        
        if not username:
            self.status_label.config(text="Username required", fg="#e74c3c")
            return
        
        if not os.path.exists(self.audio_file):
            self.status_label.config(text="Please record voice sample", fg="#e74c3c")
            return
        
        try:
            # Create embeddings directory if not exists
            os.makedirs(p.EMBED_LIST_FILE, exist_ok=True)
            
            # Check if user already exists
            user_file = os.path.join(p.EMBED_LIST_FILE, f"{username}.npy")
            if os.path.exists(user_file):
                self.status_label.config(text="Username already exists", fg="#e74c3c")
                return
            
            # Enroll user
            enroll(username, self.audio_file)
            self.status_label.config(text="Signup successful! Please login", fg="#2ecc71")
            
            # Clear fields
            self.username_entry.delete(0, tk.END)
            for widget in self.spectrogram_frame.winfo_children():
                widget.destroy()
            
            # Delete temp file
            if os.path.exists(self.audio_file):
                os.remove(self.audio_file)
            
            # Go to login page
            self.controller.show_page("VoiceLoginPage")
        except Exception as e:
            self.status_label.config(text=f"Signup failed: {str(e)}", fg="#e74c3c")

class MainPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#f0f2f5")
        self.controller = controller
        
        # Header
        header_frame = tk.Frame(self, bg="#2c3e50")
        header_frame.pack(fill="x", pady=(0,20))
        
        tk.Label(header_frame, text="Voice Authentication System", font=("Arial", 20, "bold"), 
                bg="#2c3e50", fg="white").pack(side="left", padx=20, pady=10)
        
        logout_btn = tk.Button(header_frame, text="Logout", command=self.logout,
                             bg="#e74c3c", fg="white", bd=0, padx=10)
        logout_btn.pack(side="right", padx=20)
        
        # Main Content
        main_frame = tk.Frame(self, bg="#f0f2f5")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Left Panel - Enrollment
        left_frame = tk.LabelFrame(main_frame, text="User Enrollment", 
                                 font=("Arial", 14), bg="#f0f2f5", padx=10, pady=10)
        left_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Name Entry
        tk.Label(left_frame, text="User Name:", bg="#f0f2f5").pack(pady=(0,5))
        self.name_entry = tk.Entry(left_frame, width=30)
        self.name_entry.pack(pady=(0,15))
        
        # Audio Selection
        tk.Button(left_frame, text="Select Audio File", command=self.select_enroll_file,
                bg="#3498db", fg="white").pack(pady=5)
        self.enroll_file_label = tk.Label(left_frame, text="No file selected", 
                                        bg="#f0f2f5", wraplength=250)
        self.enroll_file_label.pack(pady=5)
        
        # Spectrogram Preview
        self.enroll_spectrogram_frame = tk.Frame(left_frame, bg="white", height=150)
        self.enroll_spectrogram_frame.pack(fill="x", pady=10)
        
        # Enroll Button
        tk.Button(left_frame, text="Enroll User", command=self.enroll_user,
                bg="#2ecc71", fg="white").pack(pady=10)
        
        # Right Panel - Recognition
        right_frame = tk.LabelFrame(main_frame, text="User Recognition", 
                                  font=("Arial", 14), bg="#f0f2f5", padx=10, pady=10)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Audio Selection
        tk.Button(right_frame, text="Select Audio File", command=self.select_recognize_file,
                bg="#3498db", fg="white").pack(pady=5)
        self.recognize_file_label = tk.Label(right_frame, text="No file selected", 
                                           bg="#f0f2f5", wraplength=250)
        self.recognize_file_label.pack(pady=5)
        
        # Spectrogram Preview
        self.recognize_spectrogram_frame = tk.Frame(right_frame, bg="white", height=150)
        self.recognize_spectrogram_frame.pack(fill="x", pady=10)
        
        # Recognize Button
        tk.Button(right_frame, text="Recognize User", command=self.recognize_user,
                bg="#e74c3c", fg="black").pack(pady=10)
        #######
        # Enrolled Users List
        users_frame = tk.LabelFrame(right_frame, text="Enrolled Users", 
                                  font=("Arial", 12), bg="#f0f2f5")
        users_frame.pack(fill="both", expand=True, pady=(20,0))
        
        self.users_listbox = tk.Listbox(users_frame, height=5)
        self.users_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Console
        console_frame = tk.LabelFrame(self, text="System Console", 
                                    font=("Arial", 12), bg="#f0f2f5")
        console_frame.pack(fill="both", expand=True, padx=20, pady=(0,20))
        
        self.console = tk.Text(console_frame, height=8, state="disabled",
                             bg="#ecf0f1", font=("Consolas", 10))
        self.console.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(console_frame, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=(0,5))
        
        # Initialize variables
        self.enroll_file = None
        self.recognize_file = None
    
    def logout(self):
        self.controller.show_page("VoiceLoginPage")
    
    def refresh_enrolled_users(self):
        self.users_listbox.delete(0, tk.END)
        if os.path.exists(p.EMBED_LIST_FILE):
            for file in os.listdir(p.EMBED_LIST_FILE):
                if file.endswith(".npy"):
                    self.users_listbox.insert(tk.END, file[:-4])
    
    def log_message(self, message):
        self.console.config(state="normal")
        self.console.insert("end", message + "\n")
        self.console.see("end")
        self.console.config(state="disabled")
    
    def show_spectrogram(self, filepath, target_frame):
        for widget in target_frame.winfo_children():
            widget.destroy()
        
        try:
            buckets_var = buckets(p.MAX_SEC, p.BUCKET_STEP, p.FRAME_STEP)
            spectrogram = get_fft_spectrum(filepath, buckets_var)
            
            fig = plt.Figure(figsize=(5, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            cax = ax.imshow(spectrogram, aspect='auto', origin='lower')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Spectrogram")
            ax.axis('off')
            
            canvas = FigureCanvasTkAgg(fig, master=target_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            self.log_message(f"Spectrogram error: {str(e)}")
    
    def select_enroll_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac")])
        if filepath:
            self.enroll_file = filepath
            self.enroll_file_label.config(text=os.path.basename(filepath))
            self.show_spectrogram(filepath, self.enroll_spectrogram_frame)
            self.log_message(f"Selected enrollment file: {filepath}")
    
    def select_recognize_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac")])
        if filepath:
            self.recognize_file = filepath
            self.recognize_file_label.config(text=os.path.basename(filepath))
            self.show_spectrogram(filepath, self.recognize_spectrogram_frame)
            self.log_message(f"Selected recognition file: {filepath}")
    
    def enroll_user(self):
        if not hasattr(self, 'enroll_file'):
            messagebox.showerror("Error", "Please select an audio file first")
            return
        
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
        
        self.progress["value"] = 30
        self.root.update()
        
        self.log_message(f"Enrolling user: {name}")
        try:
            enroll(name, self.enroll_file)
            self.progress["value"] = 100
            self.log_message("Enrollment successful!")
            self.refresh_enrolled_users()
            self.name_entry.delete(0, tk.END)
            self.enroll_file_label.config(text="No file selected")
            for widget in self.enroll_spectrogram_frame.winfo_children():
                widget.destroy()
            del self.enroll_file
        except Exception as e:
            self.progress["value"] = 0
            self.log_message(f"Enrollment failed: {str(e)}")
            messagebox.showerror("Error", f"Enrollment failed: {str(e)}")
    
    def recognize_user(self):
        if not hasattr(self, 'recognize_file'):
            messagebox.showerror("Error", "Please select an audio file first")
            return
        
        self.progress["value"] = 30
        self.root.update()
        
        self.log_message("Starting recognition...")
        try:
            recognize(self.recognize_file)
            self.progress["value"] = 100
        except Exception as e:
            self.progress["value"] = 0
            self.log_message(f"Recognition failed: {str(e)}")
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAuthApp(root)
    root.mainloop()