import tkinter as tk
from tkinter import messagebox
import subprocess
import sys

class StartPage:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Recognition Game - Start Page")

        # Set up the frame
        self.frame = tk.Frame(root)
        self.frame.pack(pady=20, padx=20)

        # Title label
        self.title_label = tk.Label(self.frame, text="Welcome to Object Recognition Game!", font=("Arial", 20))
        self.title_label.pack(pady=10)

        # Start button
        self.start_button = tk.Button(self.frame, text="Start Game", command=self.start_game, font=("Arial", 16))
        self.start_button.pack(pady=10)

        # Exit button
        self.exit_button = tk.Button(self.frame, text="Exit", command=root.quit, font=("Arial", 16))
        self.exit_button.pack(pady=10)

    def start_game(self):
        """Launch the Game.py script and close the start page."""
        try:
            # Close the start page window
            self.root.destroy()

            # Launch the Game.py script
            subprocess.run([sys.executable, 'Webcam.py'], check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred while starting the game: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StartPage(root)
    root.mainloop()
