import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "predictions.csv"
OUTPUT_PATH = BASE_DIR / "prediction_week.png"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

with INPUT_PATH.open(newline="", encoding="utf-8-sig") as input_file:
    rows = list(csv.DictReader(input_file))

for row in rows:
    row["target_datetime"] = datetime.strptime(row["target_time"], TIME_FORMAT)
    row["actual"] = float(row["actual_wind_speed_ms"])
    row["baseline"] = float(row["current_wind_speed_prediction_ms"])
    row["regression"] = float(row["linear_regression_prediction_ms"])

start = datetime(2024, 12, 1, 0, 0, 0)
end = datetime(2024, 12, 7, 23, 0, 0)
week_rows = [row for row in rows if start <= row["target_datetime"] <= end]

all_actual = [row["actual"] for row in rows]
all_baseline = [row["baseline"] for row in rows]
all_regression = [row["regression"] for row in rows]
baseline_mae = sum(abs(prediction - actual) for prediction, actual in zip(all_baseline, all_actual)) / len(rows)
regression_mae = sum(abs(prediction - actual) for prediction, actual in zip(all_regression, all_actual)) / len(rows)

times = [row["target_datetime"] for row in week_rows]
plt.figure(figsize=(14, 6))
plt.plot(times, [row["actual"] for row in week_rows], color="black", label="Observed")
plt.plot(
    times,
    [row["baseline"] for row in week_rows],
    color="gray",
    linestyle="--",
    label="Current wind baseline",
)
plt.plot(
    times,
    [row["regression"] for row in week_rows],
    color="blue",
    label="Linear regression",
)
plt.xlabel("Target time (KST)")
plt.ylabel("Wind speed (m/s)")
plt.title(
    "Wind-speed predictions: 2024-12-01 to 2024-12-07 (KST)",
    loc="left",
    pad=24,
)
plt.legend(ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.16))
plt.grid(True, alpha=0.25)
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.gcf().autofmt_xdate()
plt.tight_layout(rect=(0, 0, 1, 0.88))
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()

print(f"week_rows={len(week_rows)}")
print(f"december_rows={len(rows)}")
print(f"current_wind_mae_december={baseline_mae:.10f}")
print(f"linear_regression_mae_december={regression_mae:.10f}")
print(f"output={OUTPUT_PATH}")