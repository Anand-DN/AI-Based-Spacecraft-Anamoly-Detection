import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_ROOT = Path(__file__).parent / "ESA-Mission1"
CSV_DIR = Path(__file__).parent / "data" / "csv"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"

for d in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def get_target_channels():
    meta = pd.read_csv(DATA_ROOT / "channels.csv")
    mask = (meta["Subsystem"] == "subsystem_6") & (meta["Physical Unit"] == "physical_unit_3")
    return meta[mask]["Channel"].tolist()


def load_and_merge_channels(channel_names, subset=None):
    print(f"\n[2a] Loading {len(channel_names)} channels from CSVs...")
    dfs = []
    for ch in tqdm(channel_names, desc="Loading CSVs"):
        csv_path = CSV_DIR / f"{ch}.csv"
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        dfs.append(df)
    merged = pd.concat(dfs, axis=1, join="outer")
    merged = merged.sort_index()
    nan_frac = merged.isna().sum().sum() / merged.size
    if nan_frac > 0.01:
        print(f"  Warning: {nan_frac:.2%} NaN values, forward/backward filling")
    merged = merged.ffill().bfill()
    merged = merged.astype(np.float32)
    if subset and len(merged) > subset:
        merged = merged.iloc[:subset]
        print(f"  Subset: {subset} rows")
    print(f"  Merged shape: {merged.shape}")
    print(f"  Date range: {merged.index[0]} to {merged.index[-1]}")
    return merged


def generate_labels(data_index, target_channels):
    print("\n[2b] Generating anomaly labels from labels.csv...")
    labels_df = pd.read_csv(DATA_ROOT / "labels.csv")
    labels_df["StartTime"] = pd.to_datetime(labels_df["StartTime"]).dt.tz_localize(None)
    labels_df["EndTime"] = pd.to_datetime(labels_df["EndTime"]).dt.tz_localize(None)
    target_labels = labels_df[labels_df["Channel"].isin(target_channels)]
    label_matrix = np.zeros((len(data_index), len(target_channels)), dtype=np.uint8)
    time_idx = data_index.values
    for _, row in target_labels.iterrows():
        ch = row["Channel"]
        start = row["StartTime"]
        end = row["EndTime"]
        col_idx = target_channels.index(ch)
        mask = (time_idx >= start) & (time_idx <= end)
        label_matrix[mask, col_idx] = 1
    y = label_matrix.max(axis=1).astype(np.uint8)
    n_anom = int(y.sum())
    print(f"  Anomaly ratio: {n_anom / len(y):.6%}  ({n_anom} / {len(y)})")
    return y


def normalize(df):
    print("\n[3] Normalizing with MinMaxScaler...")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns, dtype=np.float32)
    return scaled_df, scaler


def temporal_split(data, y, train_frac=0.70, val_frac=0.15):
    print(f"\n[4] Temporal split ({train_frac:.0%}/{val_frac:.0%}/{1-train_frac-val_frac:.0%})...")
    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    X_train = data.iloc[:train_end]; y_train = y[:train_end]
    X_val = data.iloc[train_end:val_end]; y_val = y[train_end:val_end]
    X_test = data.iloc[val_end:]; y_test = y[val_end:]
    print(f"  Train: {len(X_train)} ({y_train.mean():.4%} anomalies)")
    print(f"  Val:   {len(X_val)} ({y_val.mean():.4%} anomalies)")
    print(f"  Test:  {len(X_test)} ({y_test.mean():.4%} anomalies)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_windows(data, labels, window_size=50, step=25):
    print(f"\n[5] Creating sliding windows (size={window_size}, step={step})...")
    n = len(data)
    windows = []
    targets = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end])
        targets.append(labels[end - 1])
    X = np.array(windows, dtype=np.float32)
    y = np.array(targets, dtype=np.uint8)
    print(f"  Windows: {X.shape} (samples, window, channels)")
    print(f"  Anomalies: {y.sum()} / {len(y)} ({y.mean():.4%})")
    return X, y


def focal_loss(gamma=2.0, alpha=0.75):
    import tensorflow as tf
    from tensorflow import keras
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating = tf.pow(1 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_factor * modulating * cross_entropy)
    return loss


def build_cnn(input_shape):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def build_cnn_lstm(input_shape):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def compute_class_weight(y):
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    weight_pos = n / (2 * n_pos) if n_pos > 0 else 1.0
    weight_neg = n / (2 * n_neg) if n_neg > 0 else 1.0
    return {0: weight_neg, 1: weight_pos}


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx]


def evaluate_model(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    avg_prec = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  {name} Results (threshold={threshold:.4f}):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    Avg Prec:  {avg_prec:.4f}")
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
            "F1": f1, "ROC-AUC": roc_auc, "Avg Precision": avg_prec}, cm, y_pred, y_prob


def plot_confusion_matrix(cm, name, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_roc_curves(results, save_path):
    plt.figure(figsize=(7, 5))
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_pr_curves(results, save_path):
    plt.figure(figsize=(7, 5))
    for name, (y_true, y_prob) in results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_metrics_comparison(all_metrics, save_path):
    df = pd.DataFrame(all_metrics).set_index("Model")
    metrics_to_plot = ["F1", "ROC-AUC", "Precision", "Recall"]
    df[metrics_to_plot].plot(kind="bar", figsize=(8, 5), rot=0)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_training_history(history, name, save_path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("accuracy", []), label="Train Acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(f"{name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def plot_anomaly_timeline(timestamps, y_true, y_score, y_pred, threshold, name, save_path, max_points=5000):
    plt.figure(figsize=(14, 4))
    step = max(1, len(timestamps) // max_points)
    idx = slice(None, None, step)
    plt.plot(timestamps[idx], y_score[idx], label="Anomaly Score", color="blue", alpha=0.7)
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold={threshold:.3f}")
    anomaly_idx = np.where(y_pred[idx] == 1)[0]
    plt.scatter(timestamps[idx][anomaly_idx], y_score[idx][anomaly_idx],
                color="red", s=10, label="Detected Anomalies", alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Anomaly Score")
    plt.title(f"Anomaly Detection Timeline - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved {save_path}")


def map_window_predictions(pred, prob, n_timestamps, window_size, step):
    scores = np.zeros(n_timestamps)
    counts = np.zeros(n_timestamps)
    n_windows = len(pred)
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > n_timestamps:
            end = n_timestamps
            start = end - window_size
        if start < 0:
            continue
        scores[start:end] += prob[i]
        counts[start:end] += 1
    counts = np.maximum(counts, 1)
    return scores / counts, counts


def main(args):
    print("=" * 60)
    print("Spacecraft Telemetry Anomaly Detection Pipeline")
    print("=" * 60)

    target_channels = get_target_channels()
    print(f"\nTarget channels: {len(target_channels)}")
    print(", ".join(target_channels))

    data = load_and_merge_channels(target_channels, subset=args.subset)
    y = generate_labels(data.index, target_channels)

    merged_df = data.copy()
    merged_df["anomaly_label"] = y
    merged_csv = PROCESSED_DIR / "merged_data.csv"
    merged_df.to_csv(merged_csv)
    print(f"\n  Saved merged data to {merged_csv}")

    data_scaled, scaler = normalize(data)
    del data

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(
        data_scaled, y, train_frac=0.70, val_frac=0.15)

    X_train_w, y_train_w = create_windows(
        X_train.values, y_train, window_size=args.window_size, step=args.window_step)
    X_val_w, y_val_w = create_windows(
        X_val.values, y_val, window_size=args.window_size, step=args.window_step)
    X_test_w, y_test_w = create_windows(
        X_test.values, y_test, window_size=args.window_size, step=args.window_step)

    np.save(PROCESSED_DIR / "X_train.npy", X_train_w)
    np.save(PROCESSED_DIR / "y_train.npy", y_train_w)
    np.save(PROCESSED_DIR / "X_val.npy", X_val_w)
    np.save(PROCESSED_DIR / "y_val.npy", y_val_w)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_w)
    np.save(PROCESSED_DIR / "y_test.npy", y_test_w)
    print(f"\n  Saved tensors to {PROCESSED_DIR}/")

    X_test_index = X_test.index[args.window_size - 1:]

    import tensorflow as tf
    tf.random.set_seed(42)

    results_for_plots = {}
    all_metrics = []

    # ── Train CNN ──
    if "cnn" in args.models:
        print("\n" + "=" * 50)
        print("Training CNN")
        print("=" * 50)
        cnn = build_cnn((args.window_size, X_train_w.shape[2]))
        cnn.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss=focal_loss(gamma=2.0, alpha=0.75),
                    metrics=["accuracy"])
        class_weight = compute_class_weight(y_train_w)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        cnn_history = cnn.fit(
            X_train_w, y_train_w, validation_data=(X_val_w, y_val_w),
            epochs=args.epochs, batch_size=args.batch_size,
            class_weight=class_weight, callbacks=callbacks, verbose=1)
        cnn.save(MODELS_DIR / "cnn.keras")
        print(f"  Model saved to {MODELS_DIR / 'cnn.keras'}")

        plot_training_history(cnn_history, "CNN", RESULTS_DIR / "training_cnn.png")

        cnn_val_prob = cnn.predict(X_val_w, verbose=0).ravel()
        cnn_thresh = find_optimal_threshold(y_val_w, cnn_val_prob)
        print(f"  CNN optimal threshold (from val): {cnn_thresh:.4f}")

        cnn_prob = cnn.predict(X_test_w, verbose=0).ravel()
        cnn_metrics, cnn_cm, cnn_pred, _ = evaluate_model("CNN", y_test_w, cnn_prob, cnn_thresh)
        plot_confusion_matrix(cnn_cm, "CNN", RESULTS_DIR / "cm_cnn.png")
        results_for_plots["CNN"] = (y_test_w, cnn_prob)
        all_metrics.append(cnn_metrics)

        cnn_full_scores, cnn_full_counts = map_window_predictions(
            cnn_pred, cnn_prob, len(X_test), args.window_size, args.window_step)
        plot_anomaly_timeline(
            X_test.index, y_test, cnn_full_scores, (cnn_full_scores >= cnn_thresh).astype(np.uint8),
            cnn_thresh, "CNN", RESULTS_DIR / "anomaly_timeline_cnn.png")

    # ── Train CNN-LSTM ──
    if "cnn_lstm" in args.models:
        print("\n" + "=" * 50)
        print("Training CNN-LSTM")
        print("=" * 50)
        lstm = build_cnn_lstm((args.window_size, X_train_w.shape[2]))
        lstm.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                     loss=focal_loss(gamma=2.0, alpha=0.75),
                     metrics=["accuracy"])
        class_weight = compute_class_weight(y_train_w)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        lstm_history = lstm.fit(
            X_train_w, y_train_w, validation_data=(X_val_w, y_val_w),
            epochs=args.epochs, batch_size=args.batch_size,
            class_weight=class_weight, callbacks=callbacks, verbose=1)
        lstm.save(MODELS_DIR / "cnn_lstm.keras")
        print(f"  Model saved to {MODELS_DIR / 'cnn_lstm.keras'}")

        plot_training_history(lstm_history, "CNN-LSTM", RESULTS_DIR / "training_cnn_lstm.png")

        lstm_val_prob = lstm.predict(X_val_w, verbose=0).ravel()
        lstm_thresh = find_optimal_threshold(y_val_w, lstm_val_prob)
        print(f"  CNN-LSTM optimal threshold (from val): {lstm_thresh:.4f}")

        lstm_prob = lstm.predict(X_test_w, verbose=0).ravel()
        lstm_metrics, lstm_cm, lstm_pred, _ = evaluate_model("CNN-LSTM", y_test_w, lstm_prob, lstm_thresh)
        plot_confusion_matrix(lstm_cm, "CNN-LSTM", RESULTS_DIR / "cm_cnn_lstm.png")
        results_for_plots["CNN-LSTM"] = (y_test_w, lstm_prob)
        all_metrics.append(lstm_metrics)

        lstm_full_scores, lstm_full_counts = map_window_predictions(
            lstm_pred, lstm_prob, len(X_test), args.window_size, args.window_step)
        plot_anomaly_timeline(
            X_test.index, y_test, lstm_full_scores, (lstm_full_scores >= lstm_thresh).astype(np.uint8),
            lstm_thresh, "CNN-LSTM", RESULTS_DIR / "anomaly_timeline_cnn_lstm.png")

    # ── Comparison plots ──
    if results_for_plots:
        print("\n" + "=" * 50)
        print("Generating comparison plots")
        print("=" * 50)
        plot_roc_curves(results_for_plots, RESULTS_DIR / "roc_curve.png")
        plot_pr_curves(results_for_plots, RESULTS_DIR / "pr_curve.png")
        if all_metrics:
            plot_metrics_comparison(all_metrics, RESULTS_DIR / "metrics_comparison.png")
            metrics_df = pd.DataFrame(all_metrics)
            metrics_csv = RESULTS_DIR / "metrics.csv"
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"  Saved metrics to {metrics_csv}")
            print("\n" + "-" * 50)
            print("Final Metrics:")
            print(metrics_df.to_string(index=False))
            print("-" * 50)
            if "CNN-LSTM" in metrics_df["Model"].values and "CNN" in metrics_df["Model"].values:
                cnn_f1 = metrics_df[metrics_df["Model"] == "CNN"]["F1"].values[0]
                lstm_f1 = metrics_df[metrics_df["Model"] == "CNN-LSTM"]["F1"].values[0]
                if lstm_f1 > cnn_f1:
                    print(f"\n  >> CNN-LSTM is the BEST model (F1={lstm_f1:.4f} vs CNN F1={cnn_f1:.4f})")
                else:
                    print(f"\n  >> CNN is the BEST model (F1={cnn_f1:.4f} vs CNN-LSTM F1={lstm_f1:.4f})")

    print("\n" + "=" * 60)
    print("Pipeline complete. Results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spacecraft Anomaly Detection Pipeline")
    parser.add_argument("--resample-sec", type=int, default=300)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-step", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=["cnn", "cnn_lstm"])
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N rows for quick testing")
    args = parser.parse_args()
    main(args)
