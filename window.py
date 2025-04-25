import threading
import tkinter as tk
from tkinter import ttk
import os
from datetime import datetime
from runner import run_cv
from main import data_loaders
import main
import shutil

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cross‑Validation Runner")
        self.geometry("600x400")

        # Dataset selector
        self.dataset_var = tk.StringVar()
        self.dataset_cb = ttk.Combobox(
            self, textvariable=self.dataset_var,
            values=['All'] + list(data_loaders.keys()), state='readonly')
        self.dataset_cb.current(0)
        self.dataset_cb.pack(pady=5)

        # Customization controls
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(pady=5)
        # initialize IntVars for spinboxes (was missing)
        self.runs_var   = tk.IntVar(value=5)
        self.folds_var  = tk.IntVar(value=5)
        self.epochs_var = tk.IntVar(value=150)

        ttk.Label(ctrl_frame, text="Runs:").grid(row=0, column=0, padx=2)
        tk.Spinbox(ctrl_frame, from_=1, to=20, textvariable=self.runs_var, width=5).grid(row=0, column=1)
        ttk.Label(ctrl_frame, text="Folds:").grid(row=0, column=2, padx=2)
        tk.Spinbox(ctrl_frame, from_=2, to=10, textvariable=self.folds_var, width=5).grid(row=0, column=3)
        ttk.Label(ctrl_frame, text="Epochs:").grid(row=0, column=4, padx=2)
        tk.Spinbox(ctrl_frame, from_=1, to=500, textvariable=self.epochs_var, width=5).grid(row=0, column=5)

        # Start button
        self.start_btn = ttk.Button(self, text="Start Cross‑Validation", command=self.start_cv)
        self.start_btn.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self, length=500, mode='determinate')
        self.progress.pack(pady=5)

        # Estimated time label
        self.time_label = ttk.Label(self, text="Estimated Time: N/A")
        self.time_label.pack(pady=5)

        # Real‑time log output
        self.log_text = tk.Text(self, height=15)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

    def start_cv(self):
        self.start_btn.config(state='disabled')
        threading.Thread(target=self._run_cv_thread, daemon=True).start()

    def _run_cv_thread(self):
        def progress_cb(done, total):
            # schedule on main thread
            def ui_update():
                pct = int(done / total * 100)
                self.progress['value'] = pct
            self.after(0, ui_update)

        def time_cb(msg):
            # schedule on main thread
            self.after(0, lambda: self.time_label.config(text=msg))

        def log_cb(msg):
            # schedule on main thread
            def ui_log():
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
            self.after(0, ui_log)
            # also append to file
            try:
                self.log_file.write(msg + "\n")
            except Exception:
                pass

        sel = self.dataset_var.get()
        if sel == 'All':
            selected = list(data_loaders.keys())
        else:
            selected = [sel]
        runs   = self.runs_var.get()
        folds  = self.folds_var.get()
        epochs = self.epochs_var.get()

        # --- new: create timestamped output dir & open log file ---
        out_base = os.path.join(os.getcwd(), "results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_base, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # --- new: create per-dataset subdirectories ---
        ds_dirs = {}
        for ds in selected:
            ds_path = os.path.join(run_dir, ds)
            os.makedirs(ds_path, exist_ok=True)
            ds_dirs[ds] = ds_path

        # redirect all plot outputs in main.py into this run directory
        main.RESULTS_DIR = run_dir  # default, will be overridden per dataset

        # --- new: generate and save initial plots for each selected dataset ---
        for ds_name in selected:
            ds_dir = ds_dirs[ds_name]
            main.RESULTS_DIR = ds_dir
            loader = data_loaders[ds_name]
            X_tr, X_val, X_te, y_tr, y_val, y_te, sr = loader()
            main.plot_fft_spectrum(X_tr[0], sr, f"{ds_name}_train_sample1", ds_dir)
            main.plot_eeg_like_stack(X_tr, y_tr, ds_name)
            main.plot_training_scatter(X_tr, y_tr, ds_name)

        # restore RESULTS_DIR to the timestamped run_dir 
        # so run_cv writes each dataset’s outputs into: results/<timestamp>/<dataset>/
        main.RESULTS_DIR = run_dir

        self.log_file = open(os.path.join(run_dir, "results.log"), "a", encoding="utf-8")
        # write header with settings
        self.log_file.write(f"Settings: dataset={selected}, runs={runs}, folds={folds}, epochs={epochs}\n")

        # show actual config
        log_cb(f"▶ Starting: runs={runs}, folds={folds}, epochs={epochs}")

        # redirect all stdout/stderr into GUI & log_file
        import sys
        class RedirectLogger:
            def write(self, msg):
                msg = msg.strip('\n')
                if msg:
                    log_cb(msg)
                    try: self._file.write(msg + "\n")
                    except: pass
            def flush(self): pass
            def __init__(self, f): self._file = f
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = RedirectLogger(self.log_file)
        sys.stderr = RedirectLogger(self.log_file)

        try:
            run_cv(progress_cb, time_cb, log_cb, selected, n_runs=runs, n_folds=folds, n_epochs=epochs)
        except Exception as e:
            log_cb(f"ERROR: {e}")
        finally:
            # restore stdout/stderr
            sys.stdout, sys.stderr = old_out, old_err
            # close log file
            try:
                self.log_file.close()
            except Exception:
                pass

            # write cumulative run summaries to cross_validation_results.txt
            src = os.path.join(run_dir, "results.log")
            dst = os.path.join(run_dir, "cross_validation_results.txt")
            try:
                # ignore invalid bytes in results.log
                with open(src, "r", encoding="utf-8", errors="ignore") as f_in, \
                     open(dst, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        if line.startswith("########## RUN") or "|" in line:
                            f_out.write(line)
                    f_out.write("✅ Complete\n")
                log_cb(f"✅ Cross-validation summary saved to: {dst}")
            except Exception as e:
                log_cb(f"Failed to write summary txt: {e}")

            # re‑enable button on main thread
            self.after(0, lambda: self.start_btn.config(state='normal'))

if __name__ == "__main__":
    App().mainloop()
