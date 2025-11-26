import os
import datetime

class ChatLogger:
    def __init__(self, log_dir="logs/chats"):
        """
        Initialize the ChatLogger.
        Creates a new log file for every session with a timestamp.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a unique filename based on the current time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.log_dir, f"session_{timestamp}.txt")
        
        # Initialize the file with a header
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"--- Chat Session Started: {timestamp} ---\n")
            f.write("------------------------------------------\n\n")
        
        print(f"[System] Logging chat to: {self.filename}")

    def log_interaction(self, speaker, text):
        """
        Logs a single interaction (User or Bot) to the file.
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {speaker}: {text}\n"
        
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            print(f"[Error] Could not write to log file: {e}")

    def log_system(self, text):
        """
        Logs system messages (e.g., Risk assessment results).
        """
        entry = f"\n[SYSTEM]: {text}\n\n"
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            print(f"[Error] Could not write to log file: {e}")