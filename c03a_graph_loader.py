import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def parse_000(file_path):
    """
    Parse a .000 file (ASCII or binary).
    Assumes either:
    - Text file: lines with (trial, channel, sample, value) format
    - Binary file: flat float32 array (e.g. square image-like data)
    """
    try:
        raw = np.genfromtxt(file_path, comments='#', dtype=float)
        if raw.ndim == 2 and raw.shape[1] >= 4:
            values = raw[:, 3]
        else:
            values = raw.flatten()
    except Exception:
        values = np.fromfile(file_path, dtype=np.float32)
    
    n = int(np.sqrt(values.size))
    if n * n != values.size:
        raise ValueError(f"Cannot reshape: file {file_path} size {values.size} not a perfect square")
    
    return values.reshape(n, n)

def get_label_from_filename(path):
    """
    Extracts label from filename.
    Assumes format like: 'c03a_1.000' → label 1
    Adjust if your filenames are different!
    """
    base = os.path.basename(path)
    name_part = base.split('.')[0]
    if '_' in name_part:
        return int(name_part.split('_')[1])
    else:
        return 0  # default label if not present

def load_c03a_graph(
    folder_path,
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """
    Loads graphs from .000 files.
    Returns: (X_train, X_val, X_test, y_train, y_val, y_test, sampling_rate)
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    files = sorted(glob.glob(os.path.join(folder_path, "*.000")))
    if not files:
        raise ValueError(f"No .000 files found in: {folder_path}")

    graphs = [parse_000(f) for f in files]
    labels = np.array([get_label_from_filename(f) for f in files])

    graphs = np.array(graphs)

    # split train/(val+test)
    X_train, X_rem, y_train, y_rem = train_test_split(
        graphs, labels,
        test_size=(test_size + val_size),
        stratify=labels if len(np.unique(labels)) > 1 else None,
        random_state=random_state
    )
    # split val/test
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem,
        test_size=rel_val,
        stratify=y_rem if len(np.unique(y_rem)) > 1 else None,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, None  # sr=None for graphs