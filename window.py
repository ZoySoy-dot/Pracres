import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
from datetime import datetime
from runner import run_cv
import runner
import main
import shutil
from PIL import Image
import time
import numpy as np
from sklearn.model_selection import train_test_split
import tempfile
import tarfile
from functools import partial
from main import data_loaders, load_any_file, N_EEG_CHANNELS, N_ECG_CHANNELS, N_AUDIO_CHANNELS, EPOCHS
import torch

try:
    import PIL.ImageTk as ImageTk
except ImportError:
    ImageTk = None

# expose ImageTk for matplotlib’s backend import
import PIL
PIL.ImageTk = tk

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Apply theme and configure window
        style = ttk.Style()
        style.theme_use('clam')
        self.title("Cross‑Validation Runner")
        self.minsize(650, 500)
        self.columnconfigure(0, weight=1)

        # Menu bar
        self.menu_bar = tk.Menu(self)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=self.menu_bar)

        # Dataset selector
        ttk.Label(self, text="Dataset:").grid(row=0, column=0, sticky='w', padx=10, pady=(10,0))
        self.dataset_var = tk.StringVar()
        ds_keys = ["All"] + [k for k in data_loaders if k in ("EEG", "ECG", "Audio")] + ["Custom"]
        self.dataset_cb = ttk.Combobox(self, textvariable=self.dataset_var,
                                       values=ds_keys, state='readonly')
        self.dataset_cb.current(0)
        self.dataset_cb.grid(row=1, column=0, sticky='we', padx=10)
        
        # Model selector
        ttk.Label(self, text="Model:").grid(row=0, column=2, sticky='w', padx=10, pady=(10,0))
        self.model_var = tk.StringVar()
        model_options = ["All models", "SNN_V2", "FENS_MLP_V4", "SCOFNA_Conv_V4"]
        self.model_cb = ttk.Combobox(self, textvariable=self.model_var, values=model_options, state='readonly')
        self.model_cb.current(0)
        self.model_cb.grid(row=1, column=2, sticky='we', padx=10)


        # Disclosure Label
        self.disclosure_label = ttk.Label(self, text="", font=("Arial", 9, "italic"), foreground="gray")
        self.disclosure_label.grid(row=2, column=0, sticky='w', padx=10, pady=(0,5))
        self.dataset_cb.bind("<<ComboboxSelected>>", self.update_disclosure)

        # Upload button
        self.upload_btn = ttk.Button(self, text="Upload Dataset…", command=self.on_browse)
        self.upload_btn.grid(row=1, column=1, sticky='w', padx=(5,0))
        self.upload_filename_label = ttk.Label(self, text="", font=("Arial", 9))
        self.upload_filename_label.grid(row=2, column=1, sticky='w', padx=(5,0))

        # Settings frame
        settings = ttk.LabelFrame(self, text="Settings")
        settings.grid(row=3, column=0, sticky='we', padx=10, pady=5, columnspan=2)
        for i in range(6): settings.columnconfigure(i, weight=1)

        self.runs_var   = tk.IntVar(value=5)
        self.folds_var  = tk.IntVar(value=5)
        self.epochs_var = tk.IntVar(value=150)
        self.samples_var = tk.IntVar(value=0)
        ttk.Label(settings, text="Runs:").grid(row=0, column=0, sticky='e')
        tk.Spinbox(settings, from_=1, to=20, textvariable=self.runs_var, width=5).grid(row=0, column=1)
        ttk.Label(settings, text="Folds:").grid(row=0, column=2, sticky='e')
        tk.Spinbox(settings, from_=2, to=10, textvariable=self.folds_var, width=5).grid(row=0, column=3)
        ttk.Label(settings, text="Epochs:").grid(row=0, column=4, sticky='e')
        tk.Spinbox(settings, from_=1, to=500, textvariable=self.epochs_var, width=5).grid(row=0, column=5)
        ttk.Label(settings, text="Samples:").grid(row=1, column=0, sticky='e')
        tk.Spinbox(settings, from_=0, to=100000, textvariable=self.samples_var, width=10, state='readonly').grid(row=1, column=1)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=4, column=0, sticky='w', padx=10, pady=5, columnspan=2)
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self.start_training)
        self.start_btn.pack(side='left', padx=(0,5))
        self.stop_event = threading.Event()
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.on_stop, state='disabled')
        self.stop_btn.pack(side='left')

        # Device selector (hide GPU option if no CUDA)
        ttk.Label(self, text="Device:").grid(row=0, column=3, sticky='w', padx=10, pady=(10,0))
        self.device_var = tk.StringVar()
        # only offer CUDA if it's actually available
        if torch.cuda.is_available():
            device_options = ["cuda", "cpu"]
        else:
            device_options = ["cpu"]
        self.device_cb = ttk.Combobox(
            self, textvariable=self.device_var,
            values=device_options, state='readonly'
        )
        # default to first (cpu when no GPU, else cuda)
        self.device_cb.current(0)
        self.device_cb.grid(row=1, column=3, sticky='we', padx=10)

        # Separator
        ttk.Separator(self, orient='horizontal').grid(row=5, column=0, sticky='ew', padx=10, pady=5, columnspan=2)

        # Progress & ETR
        prog_frame = ttk.Frame(self)
        prog_frame.grid(row=6, column=0, sticky='we', padx=10, columnspan=2)
        prog_frame.columnconfigure(1, weight=1)
        ttk.Label(prog_frame, text="Progress:").grid(row=0, column=0, sticky='w')
        self.progress = ttk.Progressbar(prog_frame, length=100, mode='determinate')
        self.progress.grid(row=0, column=1, sticky='we', padx=(5,0))
        self.time_label = ttk.Label(self, text="Estimated Time: N/A")
        self.time_label.grid(row=7, column=0, sticky='w', padx=10, pady=(5,0), columnspan=2)

        # Tabs: Log, Results
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=8, column=0, sticky='nsew', padx=10, pady=(0,10), columnspan=2)
        self.rowconfigure(8, weight=1)

        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")
        self.log_text = scrolledtext.ScrolledText(log_frame)
        self.log_text.pack(fill='both', expand=True)

        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        self.results_text = scrolledtext.ScrolledText(results_frame)
        self.results_text.pack(fill='both', expand=True)

    def update_disclosure(self, event=None):
        selected = self.dataset_var.get()
        if selected == "Custom" or selected == "All":
            self.disclosure_label.config(text="")
        else:
            self.disclosure_label.config(text="⚡ Using synthetic dataset.")

    def on_stop(self):
        self.stop_event.set()

    def on_browse(self):
        path = filedialog.askopenfilename(
            title="Select dataset file",
            filetypes=[("Dataset files", "*.csv *.edf *.mat *.txt *.zip *.tar.gz *.tar *.000"), ("All files", "*.*")]
        )
        if not path:
            return

        self.upload_filename_label.config(text=os.path.basename(path))

        ext = os.path.splitext(path)[-1].lower()
        if ext in (".gz", ".tar", ".tgz") or path.endswith(".tar.gz"):
            tmpdir = tempfile.mkdtemp()
            try:
                with tarfile.open(path, 'r:gz') as tar:
                    tar.extractall(tmpdir)
                usable_files = [os.path.join(root, f) for root, _, files in os.walk(tmpdir) for f in files]
                if not usable_files:
                    self._log("⚠ No usable files found inside archive.")
                    return
                path = usable_files[0]
            except Exception as e:
                self._log(f"❌ Failed to extract: {e}")
                return

        try:
            X, _, _, _, _, _, _ = load_any_file(path)
            n_samples = X.shape[0]
            ch = X.shape[1] if X.ndim > 1 else 1
            if ch == N_AUDIO_CHANNELS:
                typ = "Audio"
            elif ch == N_EEG_CHANNELS:
                typ = "EEG"
            elif ch == N_ECG_CHANNELS:
                typ = "ECG"
            else:
                typ = "Custom"
        except Exception:
            typ, n_samples = "Custom", 0

        vals = list(self.dataset_cb['values'])
        if "Custom" not in vals:
            vals.append("Custom")
            self.dataset_cb['values'] = vals
        self.dataset_cb.set(typ)

        for key in list(data_loaders.keys()):
            data_loaders[key] = partial(load_any_file, filepath=path)

        self.samples_var.set(n_samples)
        self.epochs_var.set(EPOCHS)

    def start_training(self):
        ds = self.dataset_var.get()
        self._progress_init = False
        self.progress.config(value=0)
        self.time_label.config(text="Estimated Time: N/A")
        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        runner.stop_event = self.stop_event
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        ds_sel = self.dataset_var.get()
        runs  = self.runs_var.get()
        folds = self.folds_var.get()
        epochs= self.epochs_var.get()

        self._log(f"▶ Running {runs}×{folds}-fold CV")
        self.after(0, lambda: self.progress.config(value=0))

        # handle “All models” → None
        model_choice = self.model_var.get()
        if model_choice == "All models":
            model_choice = None

        device_choice = self.device_var.get()

        # prepare list of datasets to run
        if ds_sel == "All":
            datasets = list(data_loaders.keys())
        else:
            datasets = [ds_sel]

        for ds in datasets:
            self._log(f"▶ Starting CV on {ds} for {epochs} epochs")
            try:
                run_cv(ds, runs, folds, epochs,
                       progress_cb=self._update_progress,
                       time_cb=self._update_time,
                       log_cb=self._log,
                       model=model_choice,
                       device=device_choice)
                self._log(f"✅ Completed CV on {ds}")
            except Exception as e:
                self._log(f"❌ Error on {ds}: {e}")
                import traceback; self._log(traceback.format_exc())

        # restore buttons
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _update_progress(self, done, total):
        if not getattr(self, "_progress_init", False):
            self._progress_init = True
            self._total_steps = total
            self._start_time = time.time()
        done_clamped = min(done, self._total_steps)
        pct = (done_clamped / self._total_steps * 100) if self._total_steps > 0 else 0
        elapsed = time.time() - self._start_time
        rem_sec = int(elapsed / done_clamped * (self._total_steps - done_clamped)) if done_clamped > 0 else 0
        m, s = divmod(rem_sec, 60)
        self.after(0, lambda: self.progress.config(value=pct))
        self.after(0, lambda: self.time_label.config(text=f"Estimated Time: {m:02d}m{s:02d}s"))

    def _update_time(self, txt):
        self.after(0, lambda: self.time_label.config(text=f"Estimated Time: {txt}"))

    def _log(self, msg):
        def append():
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.after(0, append)

    def show_about(self):
        messagebox.showinfo("About", "Cross‑Validation Runner\nVersion 1.0")

if __name__ == "__main__":
    App().mainloop()
