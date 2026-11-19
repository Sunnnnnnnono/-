from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).with_name("load_data.csv")
RESULT_FILE = Path(__file__).with_name("load_result.csv")
PLOT_FILE = Path(__file__).with_name("Stress_plot.png")
TIME_COLUMN = "time_s"
FORCE_COLUMN = "force_N"
AREA_MM2 = 100


def main():
    data = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)

    if TIME_COLUMN not in data.columns:
        raise ValueError(f"'{TIME_COLUMN}' 열을 찾을 수 없습니다.")
    if FORCE_COLUMN not in data.columns:
        raise ValueError(f"'{FORCE_COLUMN}' 열을 찾을 수 없습니다.")

    time_values = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
    force_values = pd.to_numeric(data[FORCE_COLUMN], errors="coerce")
    valid_time = time_values.notna() & np.isfinite(time_values)
    valid_force = force_values.notna() & np.isfinite(force_values)
    valid_rows = valid_time & valid_force
    excluded_count = int((~valid_rows).sum())

    for row_index in data.index[~valid_rows]:
        for column, values, is_valid in (
            (TIME_COLUMN, time_values, valid_time),
            (FORCE_COLUMN, force_values, valid_force),
        ):
            if not is_valid.loc[row_index]:
                raw_value = data.at[row_index, column]
                shown_value = "<빈칸>" if not raw_value.strip() else repr(raw_value)
                reason = "숫자가 아님" if pd.isna(values.loc[row_index]) else "유한한 숫자가 아님"
                print(f"원본 CSV {row_index + 2}행: {column}={shown_value} ({reason})")

    data = data.loc[valid_rows].copy()
    data[TIME_COLUMN] = time_values.loc[valid_rows]
    data[FORCE_COLUMN] = force_values.loc[valid_rows]
    print(f"제외한 행 수: {excluded_count}개")
    print(f"유효한 데이터 수: {len(data)}개")

    if data.empty:
        print("유효한 데이터가 없어 계산, CSV 저장, 그래프 생성을 중단합니다.")
        return

    force = data[FORCE_COLUMN]
    data["stress.MPa"] = force / AREA_MM2
    data.to_csv(RESULT_FILE, index=False)

    max_index = force.idxmax()
    max_force = force.loc[max_index]
    max_time = data.loc[max_index, TIME_COLUMN]
    max_stress_index = data["stress.MPa"].idxmax()
    max_stress = data.loc[max_stress_index, "stress.MPa"]
    max_stress_time = data.loc[max_stress_index, TIME_COLUMN]
    over_6_mpa_count = int((data["stress.MPa"] > 6).sum())

    plt.plot(
        data[TIME_COLUMN],
        data["stress.MPa"],
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
    )
    plt.annotate(
        f"MAX-STRESS: {max_stress:g} MPa",
        xy=(max_stress_time, max_stress),
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    plt.xlabel("Time(s)")
    plt.ylabel("Stress_MPa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"데이터 개수: {len(data)}개")
    print(f"6 MPa 초과 응력 데이터 개수: {over_6_mpa_count}개")
    print(f"최대 하중: {max_force:g} N")
    print(f"해당 시간: {max_time:g} s")
    print(f"최대응력: {max_stress:g} MPa")
    print(f"해당 시간: {max_stress_time:g} s")
    print(f"그래프 저장: {PLOT_FILE.name}")


if __name__ == "__main__":
    main()