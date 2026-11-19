import csv
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "wind_dataset.csv"
OUTPUT_PATH = BASE_DIR / "wind_dataset_labeled.csv"
PREDICTION_OUTPUT_PATH = BASE_DIR / "classification_predictions.csv"
SCORE_OUTPUT_PATH = BASE_DIR / "classification_scores.csv"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
THRESHOLD_MS = 4.0
FEATURE_FIELDS = [
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_1h_before_ms",
    "wind_speed_2h_before_ms",
]

TRAIN_START = datetime(2023, 1, 1, 0, 0, 0)
TRAIN_END = datetime(2024, 11, 30, 23, 59, 59)
EVALUATION_START = datetime(2024, 12, 1, 0, 0, 0)
EVALUATION_END = datetime(2024, 12, 31, 23, 59, 59)


def parse_time(value):
    return datetime.strptime(value.strip(), TIME_FORMAT)


def find_next_wind_field(fieldnames):
    if "wind_next1h" in fieldnames:
        return "wind_next1h"
    if "target_wind_speed_ms" in fieldnames:
        return "target_wind_speed_ms"
    raise ValueError(
        "wind_next1h 또는 target_wind_speed_ms 컬럼이 필요합니다."
    )


def validate_hourly_times(rows):
    for index, row in enumerate(rows):
        prediction_time = parse_time(row["prediction_time"])
        target_time = parse_time(row["target_time"])
        if target_time - prediction_time != timedelta(hours=1):
            raise ValueError(
                f"{index + 2}행: prediction_time과 target_time의 차이가 "
                f"정확히 1시간이 아닙니다."
            )


def print_class_counts(name, rows):
    counts = Counter(int(row["label"]) for row in rows)
    print(f"{name}_rows={len(rows)}")
    print(f"{name}_0={counts[0]}")
    print(f"{name}_1={counts[1]}")


def fit_standardization(rows):
    means = []
    standard_deviations = []
    for field in FEATURE_FIELDS:
        values = [float(row[field]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        standard_deviations.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return means, standard_deviations


def standardized_features(row, means, standard_deviations):
    features = [1.0]
    for field, mean, standard_deviation in zip(
        FEATURE_FIELDS, means, standard_deviations
    ):
        features.append((float(row[field]) - mean) / standard_deviation)
    return features


def sigmoid(value):
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_logistic_regression(rows, means, standard_deviations):
    feature_rows = [
        standardized_features(row, means, standard_deviations) for row in rows
    ]
    targets = [int(row["label"]) for row in rows]
    coefficients = [0.0] * len(feature_rows[0])

    learning_rate = 0.1
    iterations = 3000
    for _ in range(iterations):
        gradients = [0.0] * len(coefficients)
        for features, target in zip(feature_rows, targets):
            probability = sigmoid(
                sum(coefficient * value for coefficient, value in zip(coefficients, features))
            )
            error = probability - target
            for index, value in enumerate(features):
                gradients[index] += error * value
        for index in range(len(coefficients)):
            coefficients[index] -= learning_rate * gradients[index] / len(feature_rows)
    return coefficients


def predict_probability(coefficients, row, means, standard_deviations):
    features = standardized_features(row, means, standard_deviations)
    score = sum(coefficient * value for coefficient, value in zip(coefficients, features))
    return sigmoid(score)


def calculate_scores(prediction_rows, prediction_field):
    hits = misses = false_alarms = normal_decisions = 0
    actual_strong_count = 0

    for row in prediction_rows:
        actual = int(row["actual_label"])
        predicted = int(row[prediction_field])
        actual_strong_count += actual

        if actual == 1 and predicted == 1:
            hits += 1
        elif actual == 1 and predicted == 0:
            misses += 1
        elif actual == 0 and predicted == 1:
            false_alarms += 1
        else:
            normal_decisions += 1

    total = len(prediction_rows)
    precision_predictions = hits + false_alarms
    precision = (
        hits / precision_predictions if precision_predictions > 0 else None
    )
    recall = hits / actual_strong_count if actual_strong_count > 0 else None

    return {
        "evaluation_rows": total,
        "actual_strong": actual_strong_count,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "normal_decisions": normal_decisions,
        "accuracy": (hits + normal_decisions) / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
    }


def create_december_plot(prediction_rows):
    plot_start = datetime(2024, 12, 1, 0, 0, 0)
    plot_end = datetime(2024, 12, 7, 23, 59, 59)
    filtered_rows = [
        row
        for row in prediction_rows
        if plot_start <= parse_time(row["target_time"]) <= plot_end
    ]
    times = [parse_time(row["target_time"]) for row in filtered_rows]

    fig, (wind_axis, prediction_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    actual_wind = [float(row["actual_wind_speed_ms"]) for row in filtered_rows]
    wind_axis.plot(times, actual_wind, color="tab:blue", marker="o", markersize=2.5)
    wind_axis.axhline(
        THRESHOLD_MS,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="강풍 기준선 4.0 m/s",
    )
    wind_axis.set_ylabel("실제 풍속 (m/s)")
    wind_axis.set_title("2024년 12월 1~7일 풍속과 강풍 판정")
    wind_axis.legend(loc="upper right")
    wind_axis.grid(True, alpha=0.3)

    actual_times = [
        time
        for time, row in zip(times, filtered_rows)
        if row["actual_label"] == "1"
    ]
    prediction050_times = [
        time
        for time, row in zip(times, filtered_rows)
        if row["logistic_prediction"] == "1"
    ]
    prediction020_times = [
        time
        for time, row in zip(times, filtered_rows)
        if row["logistic_prediction_020"] == "1"
    ]

    prediction_axis.scatter(
        actual_times,
        [3] * len(actual_times),
        color="black",
        marker="o",
        s=40,
        label="실제 강풍",
    )
    prediction_axis.scatter(
        prediction050_times,
        [2] * len(prediction050_times),
        color="tab:orange",
        marker="s",
        s=38,
        label="로지스틱 예측 0.50",
    )
    prediction_axis.scatter(
        prediction020_times,
        [1] * len(prediction020_times),
        color="tab:green",
        marker="^",
        s=38,
        label="로지스틱 예측 0.20",
    )
    prediction_axis.set_yticks([1, 2, 3])
    prediction_axis.set_yticklabels(
        ["0.20 예측", "0.50 예측", "실제 강풍"]
    )
    prediction_axis.set_xlabel("정답시각 (KST)")
    prediction_axis.legend(loc="upper right")
    prediction_axis.grid(True, alpha=0.3)

    prediction_axis.xaxis.set_major_locator(mdates.DayLocator())
    prediction_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.savefig(BASE_DIR / "strong_wind_december.png", dpi=180)
    plt.close(fig)

    december2_mistake = next(
        (
            row
            for row in filtered_rows
            if row["target_time"].startswith("2024-12-02 ")
            and row["actual_label"] == "1"
            and (
                row["logistic_prediction"] != row["actual_label"]
                or row["logistic_prediction_020"] != row["actual_label"]
            )
        ),
        None,
    )
    return filtered_rows, december2_mistake


def main():
    with INPUT_PATH.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        next_wind_field = find_next_wind_field(fieldnames)
        rows = list(reader)

    rows.sort(key=lambda row: parse_time(row["target_time"]))

    for row in rows:
        row["label"] = "1" if float(row[next_wind_field]) >= THRESHOLD_MS else "0"

    validate_hourly_times(rows)

    train_rows = [
        row
        for row in rows
        if TRAIN_START <= parse_time(row["target_time"]) <= TRAIN_END
    ]
    evaluation_rows = [
        row
        for row in rows
        if EVALUATION_START <= parse_time(row["target_time"]) <= EVALUATION_END
    ]

    output_fieldnames = fieldnames + ["label"] if "label" not in fieldnames else fieldnames
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"timezone=KST")
    print(f"label_source={next_wind_field}")
    print(f"threshold_ms={THRESHOLD_MS:.1f}")
    print_class_counts("train", train_rows)
    print_class_counts("evaluation", evaluation_rows)
    print("first_evaluation_row:")
    if evaluation_rows:
        print(evaluation_rows[0])
    else:
        print("평가 데이터가 없습니다.")
    print(f"hourly_validation=PASS")
    print(f"labeled_output={OUTPUT_PATH}")

    means, standard_deviations = fit_standardization(train_rows)
    coefficients = fit_logistic_regression(
        train_rows, means, standard_deviations
    )

    prediction_fieldnames = [
        "prediction_time",
        "target_time",
        "actual_wind_speed_ms",
        "actual_label",
        "current_wind_speed_ms",
        "always_zero_prediction",
        "current_wind_threshold_prediction",
        "strong_wind_probability",
        "logistic_prediction",
        "logistic_prediction_020",
    ]
    prediction_rows = []
    for row in evaluation_rows:
        actual_wind_speed = float(row[next_wind_field])
        current_wind_speed = float(row["wind_speed_ms"])
        probability = predict_probability(
            coefficients, row, means, standard_deviations
        )
        prediction_rows.append(
            {
                "prediction_time": row["prediction_time"],
                "target_time": row["target_time"],
                "actual_wind_speed_ms": f"{actual_wind_speed:.1f}",
                "actual_label": row["label"],
                "current_wind_speed_ms": f"{current_wind_speed:.1f}",
                "always_zero_prediction": "0",
                "current_wind_threshold_prediction": (
                    "1" if current_wind_speed >= THRESHOLD_MS else "0"
                ),
                "strong_wind_probability": f"{probability:.10f}",
                "logistic_prediction": "1" if probability >= 0.50 else "0",
                "logistic_prediction_020": "1" if probability >= 0.20 else "0",
            }
        )

    with PREDICTION_OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=prediction_fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"prediction_output={PREDICTION_OUTPUT_PATH}")
    print("first_5_prediction_rows:")
    for prediction_row in prediction_rows[:5]:
        print(prediction_row)

    evaluation_target_times = [
        row["target_time"] for row in evaluation_rows
    ]
    prediction_target_times = [
        row["target_time"] for row in prediction_rows
    ]
    if len(prediction_rows) != len(evaluation_rows):
        raise ValueError("예측 행수와 평가 행수가 일치하지 않습니다.")
    if prediction_target_times != evaluation_target_times:
        raise ValueError("예측 행과 평가 행의 시각 집합 또는 순서가 일치하지 않습니다.")

    methods = [
        ("always_zero", "always_zero_prediction"),
        ("current_wind_4plus", "current_wind_threshold_prediction"),
        ("logistic_050", "logistic_prediction"),
        ("logistic_020", "logistic_prediction_020"),
    ]
    score_rows = []
    for method_name, prediction_field in methods:
        scores = calculate_scores(prediction_rows, prediction_field)
        score_rows.append(
            {
                "method": method_name,
                "evaluation_rows": scores["evaluation_rows"],
                "actual_strong": scores["actual_strong"],
                "hits": scores["hits"],
                "misses": scores["misses"],
                "false_alarms": scores["false_alarms"],
                "normal_decisions": scores["normal_decisions"],
                "accuracy": f"{scores['accuracy']:.6f}",
                "precision": (
                    f"{scores['precision']:.6f}"
                    if scores["precision"] is not None
                    else "해당없음"
                ),
                "recall": (
                    f"{scores['recall']:.6f}"
                    if scores["recall"] is not None
                    else "해당없음"
                ),
            }
        )

    score_fieldnames = [
        "method",
        "evaluation_rows",
        "actual_strong",
        "hits",
        "misses",
        "false_alarms",
        "normal_decisions",
        "accuracy",
        "precision",
        "recall",
    ]
    with SCORE_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as score_file:
        writer = csv.DictWriter(score_file, fieldnames=score_fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    print(f"same_evaluation_rows=PASS ({len(evaluation_rows)}행)")
    print(f"actual_strong={len([row for row in evaluation_rows if row['label'] == '1'])}")
    print("classification_scores:")
    print(
        "method,evaluation_rows,actual_strong,hits,misses,"
        "false_alarms,normal_decisions,accuracy,precision,recall"
    )
    for score_row in score_rows:
        print(",".join(str(score_row[field]) for field in score_fieldnames))
    print(f"score_output={SCORE_OUTPUT_PATH}")

    december_rows, december2_mistake = create_december_plot(prediction_rows)
    print(f"december_1_to_7_rows={len(december_rows)}")
    print(f"plot_output={BASE_DIR / 'strong_wind_december.png'}")
    print("december_2_mistake:")
    if december2_mistake is None:
        print("틀린 사례가 없습니다.")
    else:
        print(december2_mistake)


if __name__ == "__main__":
    main()
