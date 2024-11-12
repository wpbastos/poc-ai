# src/gui/chat_window.py
import customtkinter as ctk
from typing import Callable
import yaml

class ChatWindow:
    def __init__(self, config_path: str, send_callback: Callable):
        self.config = self._load_config(config_path)
        self.send_callback = send_callback
        
        # Initialize window
        self.window = ctk.CTk()
        self.window.title(self.config['gui']['title'])
        self.window.geometry(f"{self.config['gui']['width']}x{self.config['gui']['height']}")
        
        # Set theme
        ctk.set_appearance_mode(self.config['gui']['theme'])
        
        self._init_ui()
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _init_ui(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Chat display area
        self.chat_display = ctk.CTkTextbox(self.main_frame, height=400)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        self.chat_display.configure(state="disabled")
        
        # Input area frame
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # Message input
        self.message_input = ctk.CTkTextbox(input_frame, height=100)
        self.message_input.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Temperature slider
        self.temp_slider = ctk.CTkSlider(
            input_frame,
            orientation="vertical",
            from_=0,
            to=1,
            number_of_steps=20,
            value=self.config['llm']['temperature']
        )
        self.temp_slider.pack(side="left", padx=5)
        
        # Send button
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self._handle_send
        )
        self.send_button.pack(side="right", padx=5)
        
    def _handle_send(self):
        message = self.message_input.get("1.0", "end-1c")
        if message.strip():
            temperature = self.temp_slider.get()
            self.send_callback(message, temperature)
            self.message_input.delete("1.0", "end")
            
    def add_message(self, role: str, content: str):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"\n{role}: {content}\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
        
    def run(self):
        self.window.mainloop()