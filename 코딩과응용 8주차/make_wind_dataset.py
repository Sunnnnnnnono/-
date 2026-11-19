import csv
from datetime import datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "asos_108_2023_2024.csv"
OUTPUT_PATH = BASE_DIR / "wind_dataset.csv"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def is_present(value):
    return value is not None and value.strip() != ""


with INPUT_PATH.open(newline="", encoding="utf-8-sig") as input_file:
    source_rows = list(csv.DictReader(input_file))

hourly_rows = {}
for source_row in source_rows:
    timestamp = datetime.strptime(source_row["datetime"].strip(), TIME_FORMAT)
    if timestamp in hourly_rows:
        raise ValueError(f"중복 관측시각: {timestamp.strftime(TIME_FORMAT)}")
    hourly_rows[timestamp] = source_row

first_timestamp = min(hourly_rows)
last_timestamp = max(hourly_rows)
aligned_rows = {}
timestamp = first_timestamp
while timestamp <= last_timestamp:
    aligned_rows[timestamp] = hourly_rows.get(timestamp, {})
    timestamp += timedelta(hours=1)

output_fields = [
    "prediction_time",
    "target_time",
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_before_ms",
    "wind_speed_2h_before_ms",
    "target_wind_speed_ms",
]
dataset_rows = []
for prediction_time, current_row in aligned_rows.items():
    target_time = prediction_time + timedelta(hours=1)
    input_values = {
        "wind_speed_ms": current_row.get("wind_speed_ms", ""),
        "temperature_C": current_row.get("temperature_C", ""),
        "humidity_pct": current_row.get("humidity_pct", ""),
        "wind_speed_1h_before_ms": aligned_rows.get(
            prediction_time - timedelta(hours=1), {}
        ).get("wind_speed_ms", ""),
        "wind_speed_2h_before_ms": aligned_rows.get(
            prediction_time - timedelta(hours=2), {}
        ).get("wind_speed_ms", ""),
    }
    target_wind_speed = aligned_rows.get(target_time, {}).get("wind_speed_ms", "")
    if not all(is_present(value) for value in input_values.values()):
        continue
    if not is_present(target_wind_speed):
        continue

    dataset_rows.append(
        {
            "prediction_time": prediction_time.strftime(TIME_FORMAT),
            "target_time": target_time.strftime(TIME_FORMAT),
            **input_values,
            "target_wind_speed_ms": target_wind_speed,
        }
    )

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(dataset_rows)

print(f"aligned_rows={len(aligned_rows)}")
print(f"saved_rows={len(dataset_rows)}")
print("first_4_rows:")
for row in dataset_rows[:4]:
    print(row)