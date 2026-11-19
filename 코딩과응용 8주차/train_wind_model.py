import csv
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "wind_dataset.csv"
OUTPUT_PATH = BASE_DIR / "predictions.csv"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURE_FIELDS = [
    "wind_speed_ms",
    "wind_speed_1h_before_ms",
    "wind_speed_2h_before_ms",
    "temperature_C",
    "humidity_pct",
]


def solve_linear_system(matrix, values):
    augmented = [row[:] + [value] for row, value in zip(matrix, values)]
    size = len(values)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("선형회귀 행렬을 풀 수 없습니다.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value_value
                for current, pivot_value_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def fit_linear_regression(rows):
    design_rows = [[1.0] + [float(row[field]) for field in FEATURE_FIELDS] for row in rows]
    targets = [float(row["target_wind_speed_ms"]) for row in rows]
    size = len(design_rows[0])
    gram_matrix = [
        [sum(row[left] * row[right] for row in design_rows) for right in range(size)]
        for left in range(size)
    ]
    gram_targets = [
        sum(row[index] * target for row, target in zip(design_rows, targets))
        for index in range(size)
    ]
    return solve_linear_system(gram_matrix, gram_targets)


def predict(coefficients, row):
    values = [1.0] + [float(row[field]) for field in FEATURE_FIELDS]
    return max(0.0, sum(coefficient * value for coefficient, value in zip(coefficients, values)))


with INPUT_PATH.open(newline="", encoding="utf-8-sig") as input_file:
    rows = list(csv.DictReader(input_file))

rows.sort(key=lambda row: datetime.strptime(row["target_time"], TIME_FORMAT))
train_end = datetime(2024, 11, 30, 23, 0, 0)
evaluation_start = datetime(2024, 12, 1, 0, 0, 0)
evaluation_end = datetime(2024, 12, 31, 23, 0, 0)
train_rows = [
    row for row in rows if datetime.strptime(row["target_time"], TIME_FORMAT) <= train_end
]
evaluation_rows = [
    row
    for row in rows
    if evaluation_start <= datetime.strptime(row["target_time"], TIME_FORMAT) <= evaluation_end
]

coefficients = fit_linear_regression(train_rows)
prediction_rows = []
baseline_errors = []
regression_errors = []
for row in evaluation_rows:
    actual = float(row["target_wind_speed_ms"])
    baseline_prediction = float(row["wind_speed_ms"])
    regression_prediction = predict(coefficients, row)
    baseline_errors.append(abs(baseline_prediction - actual))
    regression_errors.append(abs(regression_prediction - actual))
    prediction_rows.append(
        {
            "prediction_time": row["prediction_time"],
            "target_time": row["target_time"],
            "actual_wind_speed_ms": row["target_wind_speed_ms"],
            "current_wind_speed_prediction_ms": row["wind_speed_ms"],
            "linear_regression_prediction_ms": f"{regression_prediction:.10f}",
        }
    )

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
    output_fields = [
        "prediction_time",
        "target_time",
        "actual_wind_speed_ms",
        "current_wind_speed_prediction_ms",
        "linear_regression_prediction_ms",
    ]
    writer = csv.DictWriter(output_file, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(prediction_rows)

print(f"train_rows={len(train_rows)}")
print(f"evaluation_rows={len(evaluation_rows)}")
print(f"current_wind_mae={sum(baseline_errors) / len(baseline_errors):.10f}")
print(f"linear_regression_mae={sum(regression_errors) / len(regression_errors):.10f}")
print("first_evaluation_result:")
print(prediction_rows[0])