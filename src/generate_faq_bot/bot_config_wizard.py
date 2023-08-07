import tkinter as tk
from tkinter import filedialog
import json


class BotConfigWizard:
    def __init__(self, root):
        self.root = root
        self.root.title("Faq Bot Generator")

        self.create_widgets()

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(self.root, text="Faq Bot Generator", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # Bot Name Input
        bot_name_label = tk.Label(self.root, text="Bot Name:")
        bot_name_label.pack(anchor=tk.W, padx=10)
        self.bot_name_entry = tk.Entry(self.root)
        self.bot_name_entry.pack(fill=tk.X, padx=10)

        # Files Listbox
        files_label = tk.Label(self.root, text="Select Data Files:")
        files_label.pack(anchor=tk.W, padx=10)
        self.files_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.files_listbox.pack(fill=tk.BOTH, expand=True, padx=10)
        browse_button = tk.Button(self.root, text="Browse Files", command=self.browse_files)
        browse_button.pack(padx=10, pady=5)

        # Models Folder Selector
        models_folder_label = tk.Label(self.root, text="Select Models Folder:")
        models_folder_label.pack(anchor=tk.W, padx=10)
        self.models_folder_var = tk.StringVar()
        models_folder_entry = tk.Entry(self.root, textvariable=self.models_folder_var, state=tk.DISABLED)
        models_folder_entry.pack(fill=tk.X, padx=10)
        models_folder_browse_button = tk.Button(self.root, text="Browse Folder", command=self.browse_models_folder)
        models_folder_browse_button.pack(padx=10, pady=5)

        # OK and Cancel Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        ok_button = tk.Button(button_frame, text="OK", command=self.ok_button_callback)
        ok_button.pack(side=tk.LEFT, padx=10)
        cancel_button = tk.Button(button_frame, text="Cancel", command=self.cancel_button_callback)
        cancel_button.pack(side=tk.LEFT, padx=10)

    def browse_models_folder(self):
        folder_path = filedialog.askdirectory()
        self.models_folder_var.set(folder_path)

    def browse_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("All Files", "*.*")])
        for file_path in file_paths:
            self.files_listbox.insert(tk.END, file_path)

    def ok_button_callback(self):
        bot_name = self.bot_name_entry.get()
        selected_files = self.files_listbox.get(0, tk.END)
        models_folder = self.models_folder_var.get()

        # Create the data dictionary
        data = {
            "APP_NAME": bot_name,
            "DOCS_INDEX": models_folder + "/docs.index",
            "FAISS_STORE_PKL": models_folder + "/faiss_store.pkl",
            "FILES_PATHS": selected_files
        }

        # Write data to JSON file
        with open("bot_config.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        print("JSON file 'bot_config.json' created.")

        # You can perform further processing with bot_name and selected_files here
        print("Bot Name:", bot_name)
        print("Selected Files:", selected_files)

    def cancel_button_callback(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BotConfigWizard(root)
    root.mainloop()
