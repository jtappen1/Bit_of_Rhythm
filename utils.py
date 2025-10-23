from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def select_video_file():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
    )
    
    return file_path