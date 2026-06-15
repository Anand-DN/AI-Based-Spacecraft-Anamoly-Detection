import zipfile
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_ROOT = Path(__file__).parent / "ESA-Mission1"
CHANNELS_DIR = DATA_ROOT / "channels"
CHANNELS_META = DATA_ROOT / "channels.csv"
CSV_DIR = Path(__file__).parent / "data" / "csv"

CSV_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_RULE = "300s"

print("=" * 60)
print("Step 1: Extract Telemetry Channels to CSV")
print("=" * 60)

meta = pd.read_csv(CHANNELS_META)
print(f"\nTotal channels in catalog: {len(meta)}")

targets = meta[(meta["Subsystem"] == "subsystem_6") & (meta["Physical Unit"] == "physical_unit_3")].copy()
print(f"Thermal channels (subsystem_6 / physical_unit_3): {len(targets)}")

print("\n--- Thermal Channels ---")
print(f"{'Channel':<15} {'Subsystem':<15} {'Physical Unit':<20} {'Group':<6}")
print("-" * 56)
for _, row in targets.iterrows():
    print(f"{row['Channel']:<15} {row['Subsystem']:<15} {row['Physical Unit']:<20} {row['Group']:<6}")

target_names = targets["Channel"].tolist()
print(f"\nExtracting {len(target_names)} channels to CSV ({RESAMPLE_RULE} resample)...")

for ch_name in tqdm(target_names, desc="Extracting channels"):
    zip_path = CHANNELS_DIR / f"{ch_name}.zip"
    if not zip_path.exists():
        print(f"  WARNING: {zip_path} not found, skipping")
        continue

    csv_path = CSV_DIR / f"{ch_name}.csv"
    if csv_path.exists():
        continue

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(tmp)
        data_file = Path(tmp) / ch_name
        df = pd.read_pickle(data_file)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.resample(RESAMPLE_RULE).mean()
    df.columns = [ch_name]
    df.to_csv(csv_path)

print(f"\nAll channels saved to: {CSV_DIR}/")
print("Done.")
