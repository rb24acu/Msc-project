import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "raw" / "house_1" / "mains.dat"
output_file = BASE_DIR / "data" / "processed" / "energy_daily_kwh.csv"

print("Processing dataset in streaming chunks...")

chunk_size = 200000
daily_totals = {}
previous_last_row = None
chunk_number = 0

for chunk in pd.read_csv(
    input_file,
    sep=r"\s+",
    header=None,
    usecols=[0, 2],
    names=["timestamp", "power_watts"],
    chunksize=chunk_size
):
    chunk_number += 1
    print(f"Processing chunk {chunk_number}...")

    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunk["power_watts"] = pd.to_numeric(chunk["power_watts"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp", "power_watts"])

    if chunk.empty:
        continue

    # prepend last row from previous chunk so time gaps remain correct
    if previous_last_row is not None:
        chunk = pd.concat([previous_last_row, chunk], ignore_index=True)

    # sort just in case
    chunk = chunk.sort_values("timestamp").reset_index(drop=True)

    # compute time difference in seconds using numeric timestamps directly
    chunk["time_diff_seconds"] = chunk["timestamp"].diff()

    # use median positive gap for first row / weird gaps
    positive_gaps = chunk.loc[chunk["time_diff_seconds"] > 0, "time_diff_seconds"]
    median_gap = positive_gaps.median() if not positive_gaps.empty else 1.0

    chunk["time_diff_seconds"] = chunk["time_diff_seconds"].fillna(median_gap)
    chunk["time_diff_seconds"] = chunk["time_diff_seconds"].clip(lower=0, upper=median_gap * 5)

    # energy in kWh
    chunk["energy_kwh"] = (chunk["power_watts"] * chunk["time_diff_seconds"]) / 3600 / 1000

    # convert timestamp to day
    chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s")
    chunk["day"] = chunk["timestamp"].dt.floor("D")

    # if previous row was prepended, skip it so it is not double-counted
    start_index = 1 if previous_last_row is not None else 0
    chunk_to_sum = chunk.iloc[start_index:].copy()

    daily_chunk = chunk_to_sum.groupby("day")["energy_kwh"].sum()

    for day, value in daily_chunk.items():
        daily_totals[day] = daily_totals.get(day, 0.0) + float(value)

    # keep last row for next chunk
    previous_last_row = chunk[["timestamp", "power_watts"]].tail(1).copy()
    previous_last_row["timestamp"] = previous_last_row["timestamp"].astype("int64") / 1e9

daily_energy = pd.DataFrame(
    sorted(daily_totals.items()),
    columns=["timestamp", "energy_kwh"]
).set_index("timestamp")

daily_energy.to_csv(output_file)

print("\nDone.")
print("Daily dataset shape:", daily_energy.shape)
print("Saved to:", output_file)
print("\nFirst 5 rows:")
print(daily_energy.head())