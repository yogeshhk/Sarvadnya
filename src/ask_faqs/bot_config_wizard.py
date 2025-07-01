import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import os


class BotConfigWizard:
    def __init__(self, root):
        self.root = root
        self.root.title("FAQ Bot Generator")

        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(self.root, text="FAQ Bot Generator", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Bot Name Input
        tk.Label(self.root, text="Bot Name:").pack(anchor=tk.W, padx=10)
        self.bot_name_entry = tk.Entry(self.root)
        self.bot_name_entry.pack(fill=tk.X, padx=10)

        # Model Selection Dropdown
        tk.Label(self.root, text="Select Model:").pack(anchor=tk.W, padx=10)
        self.dropdown_var = tk.StringVar(value='VertexAI')
        self.dropdown = ttk.Combobox(self.root, textvariable=self.dropdown_var, values=['VertexAI', 'Llama2', 'Flan'])
        self.dropdown.pack(fill=tk.X, padx=10)

        # Files Listbox
        tk.Label(self.root, text="Select Data Files:").pack(anchor=tk.W, padx=10)
        self.files_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.files_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))
        tk.Button(self.root, text="Browse Files", command=self.browse_files).pack(padx=10, pady=5)

        # Models Folder Selector
        tk.Label(self.root, text="Select Models Folder:").pack(anchor=tk.W, padx=10)
        self.models_folder_var = tk.StringVar()
        models_folder_entry = tk.Entry(self.root, textvariable=self.models_folder_var, state=tk.DISABLED)
        models_folder_entry.pack(fill=tk.X, padx=10)
        tk.Button(self.root, text="Browse Folder", command=self.browse_models_folder).pack(padx=10, pady=5)

        # OK and Cancel Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=self.ok_button_callback).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.root.destroy).pack(side=tk.LEFT, padx=10)

    def browse_models_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.models_folder_var.set(folder_path)

    def browse_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("All Files", "*.*")])
        for file_path in file_paths:
            self.files_listbox.insert(tk.END, file_path)

    def ok_button_callback(self):
        bot_name = self.bot_name_entry.get().strip()
        selected_files = self.files_listbox.get(0, tk.END)
        models_folder = self.models_folder_var.get().strip()
        selected_model = self.dropdown_var.get()

        # Validation
        if not bot_name:
            messagebox.showerror("Error", "Bot name cannot be empty.")
            return
        if not models_folder:
            messagebox.showerror("Error", "Models folder not selected.")
            return
        if not selected_files:
            messagebox.showerror("Error", "No files selected.")
            return

        # Create the config dictionary
        data = {
            "APP_NAME": bot_name,
            "MODEL_NAME": selected_model,
            "DOCS_INDEX": os.path.join(models_folder, "docs.index"),
            "FAISS_STORE_PKL": os.path.join(models_folder, "faiss_store.pkl"),
            "FILES_PATHS": list(selected_files)
        }

        # Save JSON
        with open("bot_config.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        messagebox.showinfo("Success", "bot_config.json created successfully!")
        print("Bot Config Saved:", data)


if __name__ == "__main__":
    root = tk.Tk()
    app = BotConfigWizard(root)
    root.mainloop()
