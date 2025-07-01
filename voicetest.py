import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import numpy as np
import noisereduce as nr
from scipy.io.wavfile import write

class NoiseMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Noise Monitor")

        # Dropdown for device selection
        self.device_var = tk.StringVar()
        self.device_var.set("Select Input Device")
        self.devices = sd.query_devices()
        self.input_devices = [d['name'] for d in self.devices if d['max_input_channels'] > 0]
        self.device_dropdown = ttk.Combobox(root, values=self.input_devices, state="readonly")
        self.device_dropdown.set("Select Input Device")
        self.device_dropdown.pack(pady=10)

        # Gain slider
        self.gain_label = tk.Label(root, text="Gain:")
        self.gain_label.pack()
        self.gain_slider = ttk.Scale(root, from_=1, to=10, orient="horizontal", value=1)
        self.gain_slider.pack(pady=10)

        # Noise reduction checkbox
        self.noise_reduction_var = tk.BooleanVar()
        self.noise_reduction_checkbox = tk.Checkbutton(root, text="Enable Noise Reduction", variable=self.noise_reduction_var)
        self.noise_reduction_checkbox.pack(pady=10)

        # Noise level bar
        self.db_label = tk.Label(root, text="Noise Level: -∞ dB")
        self.db_label.pack()
        self.canvas = tk.Canvas(root, width=600, height=50, bg="white")
        self.canvas.pack(pady=10)
        self.bar = self.canvas.create_rectangle(0, 0, 0, 50, fill="green")

        # Start/Stop Monitoring buttons
        self.start_button = tk.Button(root, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(root, text="Stop Monitoring", command=self.stop_monitoring, state="disabled")
        self.stop_button.pack(pady=10)

        # Start/Stop Recording buttons
        self.record_button = tk.Button(root, text="Start Recording", command=self.start_recording)
        self.record_button.pack(pady=10)
        self.stop_record_button = tk.Button(root, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.stop_record_button.pack(pady=10)

        # Export button
        self.export_button = tk.Button(root, text="Export Recording", command=self.export_recording, state="disabled")
        self.export_button.pack(pady=10)

        self.stream = None
        self.running = False
        self.recording = False
        self.audio_buffer = []
        self.sample_rate = 16000

    def start_monitoring(self):
        selected_device = self.device_dropdown.get()
        if selected_device == "Select Input Device":
            messagebox.showerror("Error", "Please select an input device.")
            return

        # Get device index
        device_idx = next((i for i, d in enumerate(self.devices) if d['name'] == selected_device), None)
        if device_idx is None:
            messagebox.showerror("Error", "Invalid device selected.")
            return

        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        # Start audio stream
        self.stream = sd.InputStream(device=device_idx, channels=1, samplerate=self.sample_rate, callback=self.audio_callback)
        self.stream.start()

    def stop_monitoring(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.canvas.coords(self.bar, 0, 0, 0, 50)
        self.db_label.config(text="Noise Level: -∞ dB")

    def start_recording(self):
        if not self.running:
            messagebox.showerror("Error", "Start monitoring before recording.")
            return
        self.recording = True
        self.audio_buffer = []
        self.record_button.config(state="disabled")
        self.stop_record_button.config(state="normal")

    def stop_recording(self):
        self.recording = False
        self.record_button.config(state="normal")
        self.stop_record_button.config(state="disabled")
        self.export_button.config(state="normal")

    def export_recording(self):
        if not self.audio_buffer:
            messagebox.showerror("Error", "No recording to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if file_path:
            audio_data = np.concatenate(self.audio_buffer)
            write(file_path, self.sample_rate, (audio_data * 32767).astype(np.int16))
            messagebox.showinfo("Export", f"Recording saved to {file_path}")

    def audio_callback(self, indata, frames, time, status):
        if not self.running:
            return

        # Apply gain
        gain = self.gain_slider.get()
        indata = indata * gain

        # Apply noise reduction if enabled
        if self.noise_reduction_var.get():
            indata = nr.reduce_noise(y=indata.flatten(), sr=self.sample_rate, y_noise=indata[:self.sample_rate].flatten())

        # Calculate RMS (Root Mean Square) for noise level
        rms = np.sqrt(np.mean(indata**2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -np.inf

        # Scale RMS to fit canvas width
        noise_level = min(int(rms * 600), 600)

        # Update the noise level bar and dB label
        self.root.after(0, self.update_bar, noise_level, db)

        # Record audio if recording is enabled
        if self.recording:
            self.audio_buffer.append(indata.copy())

    def update_bar(self, noise_level, db):
        self.canvas.coords(self.bar, 0, 0, noise_level, 50)
        if noise_level > 400:
            self.canvas.itemconfig(self.bar, fill="red")
        elif noise_level > 200:
            self.canvas.itemconfig(self.bar, fill="yellow")
        else:
            self.canvas.itemconfig(self.bar, fill="green")
        self.db_label.config(text=f"Noise Level: {db:.2f} dB")

if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseMonitorApp(root)
    root.mainloop()