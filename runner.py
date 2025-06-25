import time
import numpy as np
import os
import main
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from functools import partial
from main import data_loaders, run_fold, n_splits, EPOCHS, plot_confusion_matrix, plot_comparison, plot_eeg_like_stack
from datetime import datetime

# add at module level
stop_event = None

def run_cv(dataset_name, runs, folds, epochs,
           progress_cb=None, time_cb=None, log_cb=None, graph_cb=None,
           model=None, device="cuda"):
    """
    Runs cross-validation on specified datasets, loading only once and
    reusing splits across runs/folds.
    """
    import torch
    import numpy as np
    import os
    import time
    from sklearn.model_selection import StratifiedKFold
    from datetime import datetime
    import main
    from main import data_loaders, RESULTS_DIR, plot_comparison, plot_confusion_matrix, plot_eeg_like_stack

    # 1) Load the dataset just once
    X_tr, X_val, X_te, y_tr, y_val, y_te, sr = data_loaders[dataset_name]()

    # 2) Override main.py’s device
    try:
        main.device = torch.device(device)
        main.DEVICE_NAME = device.upper()
    except:
        pass

    # 3) Build the full pools
    X_pool = np.concatenate([X_tr, X_val, X_te], axis=0)
    y_pool = np.concatenate([y_tr, y_val, y_te], axis=0).astype(int)
    X_flat = X_pool.reshape(len(y_pool), -1)

    # 4) Compute splits just once
    skf    = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    splits = list(skf.split(X_flat, y_pool))

    # 5) Setup progress bookkeeping
    runs_iter   = range(runs) if isinstance(runs, int) else runs
    total_tasks = len(runs_iter) * len(splits) * epochs
    done = 0
    start = time.perf_counter()

    def epoch_cb(epoch, total_epochs):
        nonlocal done
        done += 1
        if progress_cb:
            progress_cb(done, total_tasks)
        if time_cb:
            elapsed = time.perf_counter() - start
            avg = elapsed / done if done else 0
            rem  = int(avg * (total_tasks - done))
            m, s = divmod(rem, 60)
            time_cb(f"ETR: {m:02d}m{s:02d}s")

    # 6) Prepare accumulators
    histories_accum = {}   # run_id -> list of per‐fold histories
    test_sets       = {}   # run_id -> list of (results, conf_mats)

    # 7) Outer loops
    for run_id in runs_iter:
        run_start = time.perf_counter()
        histories_accum[run_id] = []
        if log_cb:
            log_cb(f"▶ Starting Run {run_id+1}/{len(runs_iter)}")

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if log_cb:
                log_cb(f"  • Fold {fold_idx+1}/{len(splits)}")

            res, hist, conf = _run_single_fold(
                run_id, dataset_name,
                X_pool, y_pool, sr,
                train_idx, test_idx, fold_idx,
                epoch_cb, log_cb, epochs, model, device
            )

            histories_accum[run_id].append(hist)
            test_sets.setdefault(run_id, []).append((res, conf))

            # live‐plot callback
            if graph_cb:
                accs = {m: a for m, (a,_,_) in res.items()}
                graph_cb({'run': run_id+1, 'fold': fold_idx+1, 'acc': accs})

            # per‐fold comparison
            try:
                comp_dir = os.path.join(RESULTS_DIR, dataset_name, "comparison_metrics")
                os.makedirs(comp_dir, exist_ok=True)
                prev = main.RESULTS_DIR
                main.RESULTS_DIR = comp_dir
                plot_comparison(
                    list(hist.values()),
                    list(hist.keys()),
                    f"{dataset_name}_run{run_id+1}_fold{fold_idx+1}"
                )
                main.RESULTS_DIR = prev
            except:
                pass

        # end folds
        elapsed = time.perf_counter() - run_start
        if log_cb:
            log_cb(f"▶ Run {run_id+1} done in {elapsed:.1f}s")

    # 8) Aggregate & save results/confusion
    for run_id, fold_data in test_sets.items():
        agg, conf_agg = {}, {}
        for res, conf in fold_data:
            for m, (a, r, e) in res.items():
                agg.setdefault(m, {'acc': [], 'rmse': [], 'mae': []})
                agg[m]['acc'].append(a)
                agg[m]['rmse'].append(r)
                agg[m]['mae'].append(e)
                conf_agg.setdefault(m, conf[m].copy())
                conf_agg[m] += conf[m]

        if log_cb:
            log_cb(f"▶ Summary for Run {run_id+1}:")
            for m, stats in agg.items():
                log_cb(f"    {m:<12} |"
                       f" Acc {np.mean(stats['acc']):.3f}±{np.std(stats['acc']):.3f} |"
                       f" RMSE {np.mean(stats['rmse']):.3f}±{np.std(stats['rmse']):.3f} |"
                       f" MAE {np.mean(stats['mae']):.3f}±{np.std(stats['mae']):.3f}")

        # save aggregated confusion
        for m, cm in conf_agg.items():
            prev = main.RESULTS_DIR
            main.RESULTS_DIR = os.path.join(RESULTS_DIR, dataset_name)
            plot_confusion_matrix(
                cm,
                classes=list(range(cm.shape[0])),
                title=f"{dataset_name} - {m} Confusion",
                dataset_name=dataset_name,
                model_name=m
            )
            main.RESULTS_DIR = prev

    if log_cb:
        log_cb("✅ All runs complete")

    # —————— Summary comparison across _all_ runs & folds ——————
    try:
        summary_dir = os.path.join(RESULTS_DIR, dataset_name, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        prev = main.RESULTS_DIR
        main.RESULTS_DIR = summary_dir

        # flatten out every fold‐history for every run
        combined = {}  # model -> list of history dicts
        for run_hist_list in histories_accum.values():
            for hist in run_hist_list:
                for m, h in hist.items():
                    combined.setdefault(m, []).append(h)

        avg_histories, labels = [], []
        for m, hist_list in combined.items():
            # metrics = keys of first history
            metrics = hist_list[0].keys()
            avg_h = {}
            for metric in metrics:
                # gather array of shape (n_folds*n_runs, epochs)
                arrs = [np.array(h[metric]) for h in hist_list]
                # pad to max length
                max_len = max(a.shape[0] for a in arrs)
                padded = np.stack([np.pad(a, (0, max_len - a.shape[0]), constant_values=np.nan)
                                   for a in arrs], axis=0)
                # average over axis=0
                avg_h[metric] = np.nanmean(padded, axis=0).tolist()
            avg_histories.append(avg_h)
            labels.append(m)

        plot_comparison(avg_histories, labels, f"{dataset_name}_summary")
    finally:
        main.RESULTS_DIR = prev

    # —————— Stacked‐signal plots for your final test set ——————
    try:
        stack_dir = os.path.join(RESULTS_DIR, dataset_name, "stacked_signals")
        os.makedirs(stack_dir, exist_ok=True)
        prev = main.RESULTS_DIR
        main.RESULTS_DIR = stack_dir

        plot_eeg_like_stack(X_te, y_te, dataset_name)
    finally:
        main.RESULTS_DIR = prev
        metrics_dir = os.path.join(RESULTS_DIR, dataset_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        csv_path = os.path.join(metrics_dir, "comparison_metrics.csv")
        import csv

        # aggregate across all runs+folds
        global_agg = {}
        for run_data in test_sets.values():
            for res, _ in run_data:
                for m, (a, r, e) in res.items():
                    stats = global_agg.setdefault(m, {"acc": [], "rmse": [], "mae": []})
                    stats["acc"].append(a)
                    stats["rmse"].append(r)
                    stats["mae"].append(e)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "acc_mean", "acc_std",
                "rmse_mean", "rmse_std",
                "mae_mean", "mae_std"
            ])
            for m, stats in global_agg.items():
                writer.writerow([
                    m,
                    np.mean(stats["acc"]), np.std(stats["acc"]),
                    np.mean(stats["rmse"]), np.std(stats["rmse"]),
                    np.mean(stats["mae"]), np.std(stats["mae"])
                ])
        if log_cb:
            log_cb(f"✅ Saved numeric comparison metrics to {csv_path}")
        summary_path = os.path.join(RESULTS_DIR, f"all_runs_summary_{datetime.now():%Y%m%d_%H%M%S}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Cross-Validation Summary ({datetime.now():%Y-%m-%d %H:%M:%S})\n\n")
        # For each dataset we ran
        for ds, fold_data in test_sets.items():
            f.write(f"Dataset: {ds}\n")
            # aggregate per-model
            agg = {}
            for res, _ in fold_data:
                for m, (acc, rmse, mae) in res.items():
                    agg.setdefault(m, {"acc": [], "rmse": [], "mae": []})
                    agg[m]["acc"].append(acc)
                    agg[m]["rmse"].append(rmse)
                    agg[m]["mae"].append(mae)
            # write table
            f.write(" Model           |   Acc (μ±σ)   |  RMSE (μ±σ)   |  MAE (μ±σ)\n")
            f.write("---------------------------------------------------------------\n")
            for m, stats in agg.items():
                import numpy as _np
                a_mean, a_std = _np.mean(stats["acc"]), _np.std(stats["acc"])
                r_mean, r_std = _np.mean(stats["rmse"]), _np.std(stats["rmse"])
                e_mean, e_std = _np.mean(stats["mae"]), _np.std(stats["mae"])
                f.write(f" {m:<15}| {a_mean:6.3f}±{a_std:5.3f} | {r_mean:6.3f}±{r_std:5.3f} | {e_mean:6.3f}±{e_std:5.3f}\n")
            f.write("\n")
    print(f"✅ Saved overall summary: {summary_path}")


def _run_single_fold(run_id, ds_name, X_pool, y_pool, sr,
                     train_idx, test_idx, fold_idx,
                     base_epoch_cb, log_cb, n_epochs, model, device):

    """
    Wraps run_fold to prefix each epoch log with run/dataset/fold context,
    and returns the training histories alongside results and confusion matrices.
    """
    def epoch_cb(epoch, total_epochs):
        # only invoke base if provided
        if base_epoch_cb:
            base_epoch_cb(epoch, total_epochs)

    # capture histories dict from run_fold
    results, histories, conf_mats = run_fold(
        ds_name, X_pool, y_pool, sr,
        train_idx, test_idx,
        fold_idx, epochs=n_epochs, epoch_cb=epoch_cb,
        selected_model=model, device=device  # DEVICE ADDED HERE
    )

    return results, histories, conf_mats
