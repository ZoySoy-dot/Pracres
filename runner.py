import time
import numpy as np
import os
import main
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from functools import partial
from main import data_loaders, run_fold, n_splits, EPOCHS, plot_confusion_matrix, plot_comparison, plot_eeg_like_stack

def run_cv(progress_cb, time_cb, log_cb, selected=None,
           n_runs=5, n_folds=5, n_epochs=150):
    """
    Runs cross‑validation on specified datasets.
    """
    dsets = selected or list(data_loaders.keys())
    n_dsets = len(dsets)
    total_tasks = n_runs * n_dsets * n_folds * n_epochs
    done = 0
    start_all = time.perf_counter()
    last_time = start_all
    avg_epoch = 0.0

    def epoch_cb(epoch, total_epochs):
        nonlocal done, last_time, avg_epoch
        now = time.perf_counter()
        delta = now - last_time
        last_time = now
        done += 1
        # update running average of epoch duration
        avg_epoch = delta if done == 1 else avg_epoch + (delta - avg_epoch) / done
        progress_cb(done, total_tasks)

        # compute current run/fold from overall progress
        tasks_per_dataset = n_folds * total_epochs
        tasks_per_run = len(dsets) * tasks_per_dataset
        run_idx = min(n_runs, ((done - 1) // tasks_per_run) + 1)
        fold_idx = min(n_folds, (((done - 1) % tasks_per_run) // total_epochs) + 1)

        # never allow negative remaining tasks
        rem_tasks = total_tasks - done
        rem_tasks = rem_tasks if rem_tasks > 0 else 0
        rem_sec = int(avg_epoch * rem_tasks)
        mins, secs = divmod(rem_sec, 60)
        time_cb(f"Run {run_idx}/{n_runs} Fold {fold_idx}/{n_folds} ETR: {mins:02d}:{secs:02d}")

    all_results = {}  # collect per‐dataset results
    X_test_sets, y_test_sets = {}, {}

    for run_id in range(1, n_runs+1):
        for ds_name in dsets:
            all_results[ds_name] = {}
            loader = data_loaders[ds_name]
            X_tr, X_val, X_te, y_tr, y_val, y_te, sr = loader()
            X_test_sets[ds_name] = X_te
            y_test_sets[ds_name] = y_te
            X_pool = np.concatenate([X_tr, X_val, X_te], axis=0)
            y_pool = np.concatenate([y_tr, y_val, y_te], axis=0).astype(int)
            X_flat = X_pool.reshape(len(y_pool), -1)

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(skf.split(X_flat, y_pool))
            fold_results = Parallel(n_jobs=min(len(splits), os.cpu_count()), backend='threading')(
                delayed(_run_single_fold)(
                    run_id, ds_name, X_pool, y_pool, sr,
                    train_idx, test_idx, fold_idx,
                    epoch_cb, log_cb, n_epochs
                )
                for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1)
            )
            # Aggregate per‐model metrics + confusion matrices
            agg = {}
            conf_agg = {}
            for res, conf in fold_results:
                for m, (a, r, e) in res.items():
                    agg.setdefault(m, {'acc': [], 'rmse': [], 'mae': []})
                    agg[m]['acc'].append(a)
                    agg[m]['rmse'].append(r)
                    agg[m]['mae'].append(e)
                    # sum confusion matrices
                    cm = conf[m]
                    conf_agg.setdefault(m, cm.copy())
                    conf_agg[m] += cm
            # Log run‐level summary
            log_cb(f"########## RUN {run_id} - {ds_name} ##########")
            for m, stats in agg.items():
                a_mean, a_std = np.mean(stats['acc']), np.std(stats['acc'])
                r_mean, r_std = np.mean(stats['rmse']), np.std(stats['rmse'])
                e_mean, e_std = np.mean(stats['mae']), np.std(stats['mae'])
                log_cb(f"{m:<15} | Acc: {a_mean:.4f}±{a_std:.4f} | RMSE: {r_mean:.4f}±{r_std:.4f} | MAE: {e_mean:.4f}±{e_std:.4f}")
            # now save the aggregated confusion‐matrices
            for m, cm in conf_agg.items():
                labels = list(range(cm.shape[0]))
                # ensure per-dataset folder under current RESULTS_DIR
                ds_dir = os.path.join(main.RESULTS_DIR, ds_name)
                os.makedirs(ds_dir, exist_ok=True)
                outp = os.path.join(ds_dir, f"{ds_name}_{m}_confmat.png")
                plot_confusion_matrix(cm, labels, f"{ds_name} - {m} Confusion", outp)

    # comparison charts removed to prevent attribute errors
    # only the stacked plot for test signals
    for ds_name in dsets:
        # redirect RESULTS_DIR for final stacked plots
        ds_dir = os.path.join(main.RESULTS_DIR, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        main.RESULTS_DIR = ds_dir
        plot_eeg_like_stack(
            X_test_sets[ds_name],
            y_test_sets[ds_name],
            ds_name
        )

    log_cb("✅ Complete")

# helper for parallel execution of a single fold
def _run_single_fold(run_id, ds_name, X_pool, y_pool, sr,
                     train_idx, test_idx, fold_idx,
                     base_epoch_cb, log_cb, n_epochs):
    """
    Wraps run_fold to prefix each epoch log with run/dataset/fold context.
    """
    def epoch_cb(epoch, total_epochs):
        # update progress only; suppress per-epoch messages
        base_epoch_cb(epoch, total_epochs)

    # suppressed all per‑fold logs
    results, _, conf_mats = run_fold(ds_name, X_pool, y_pool, sr,
                                     train_idx, test_idx,
                                     fold_idx, epochs=n_epochs, epoch_cb=epoch_cb)
    return results, conf_mats
