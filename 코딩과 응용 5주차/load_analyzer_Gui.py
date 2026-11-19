import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from load_analyzer import FORCE_COLUMN, TIME_COLUMN


class LoadAnalyzerGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Load Analyzer")
        self.root.geometry("900x700")
        self.path = tk.StringVar()
        self.valid_count = tk.StringVar(value="유효 데이터 수: -")
        self.section_type = tk.StringVar(value="사각형")
        self.dimensions = {
            name: tk.StringVar()
            for name in ("b", "h", "d", "D", "t", "B", "H", "tw", "tf")
        }
        self.dimension_labels = {
            "사각형": (("b", "폭 b (mm)"), ("h", "높이 h (mm)")),
            "원형봉": (("d", "직경 d (mm)"),),
            "원형관": (("D", "외경 D (mm)"), ("t", "두께 t (mm)")),
            "사각형관": (
                ("B", "외측 폭 B (mm)"),
                ("H", "외측 높이 H (mm)"),
                ("t", "두께 t (mm)"),
            ),
            "H형강": (
                ("H", "전체 높이 H (mm)"),
                ("B", "플랜지 폭 B (mm)"),
                ("tw", "웨브 두께 tw (mm)"),
                ("tf", "플랜지 두께 tf (mm)"),
            ),
        }
        self.area_text = tk.StringVar(value="단면적 A: - mm²")
        self.condition_status = tk.StringVar()
        self.input_status = tk.StringVar()
        self.result_text = tk.StringVar(value="최대 응력: - MPa | 해당 시간: - s")

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="CSV 찾아보기", command=self.select_csv).pack(
            side=tk.LEFT
        )
        ttk.Entry(controls, textvariable=self.path, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0)
        )

        section_controls = ttk.Frame(root, padding=(10, 0, 10, 10))
        section_controls.pack(fill=tk.X)
        section_select_row = ttk.Frame(section_controls)
        section_select_row.pack(fill=tk.X)
        ttk.Label(section_select_row, text="단면 종류").pack(side=tk.LEFT)
        ttk.Combobox(
            section_select_row,
            textvariable=self.section_type,
            values=tuple(self.dimension_labels),
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(6, 16))
        ttk.Button(section_select_row, text="분석", command=self.analyze).pack(
            side=tk.RIGHT
        )

        self.dimension_frame = ttk.Frame(section_controls)
        self.dimension_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(section_controls, textvariable=self.area_text).pack(
            anchor=tk.W, pady=(6, 0)
        )
        ttk.Label(section_controls, textvariable=self.condition_status).pack(
            anchor=tk.W
        )
        ttk.Label(
            section_controls, textvariable=self.input_status, foreground="red"
        ).pack(anchor=tk.W)
        ttk.Label(root, textvariable=self.valid_count, padding=(10, 0, 10, 4)).pack(
            anchor=tk.W
        )
        ttk.Label(root, textvariable=self.result_text, padding=(10, 0, 10, 10)).pack(
            anchor=tk.W
        )

        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self._set_empty_plot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas.draw()
        self.section_type.trace_add("write", self._section_changed)
        for dimension in self.dimensions.values():
            dimension.trace_add("write", self._dimensions_changed)
        self._render_dimension_fields()

    def _set_empty_plot(self):
        self.axes.clear()
        self.axes.set_xlabel("Time(s)")
        self.axes.set_ylabel("Stress(MPa)")
        self.axes.grid(True)

    def _section_changed(self, *_):
        self._render_dimension_fields()
        self._conditions_changed()

    def _dimensions_changed(self, *_):
        self._conditions_changed()

    def _conditions_changed(self):
        self._invalidate_results("조건변경됨: 다시 분석필요")
        try:
            self._calculate_area()
        except ValueError as error:
            self.input_status.set(str(error))
        else:
            self.input_status.set("")

    def _render_dimension_fields(self):
        for child in self.dimension_frame.winfo_children():
            child.destroy()
        for name, label in self.dimension_labels[self.section_type.get()]:
            field = ttk.Frame(self.dimension_frame)
            field.pack(side=tk.LEFT, padx=(0, 12))
            ttk.Label(field, text=label).pack(anchor=tk.W)
            ttk.Entry(field, textvariable=self.dimensions[name], width=10).pack(
                anchor=tk.W
            )

    def _get_dimension(self, name, label):
        raw_value = self.dimensions[name].get().strip()
        if not raw_value:
            raise ValueError(f"{label}를 입력하세요.")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"{label}는 숫자로 입력하세요.") from error
        if not np.isfinite(value):
            raise ValueError(f"{label}는 유한한 숫자로 입력하세요.")
        if value == 0:
            raise ValueError(f"{label}는 0보다 커야 합니다. (현재 입력값: 0)")
        if value < 0:
            raise ValueError(f"{label}는 음수일 수 없습니다. (현재 입력값: {value:g})")
        return value

    def _calculate_area(self):
        section = self.section_type.get()
        if section == "사각형":
            b = self._get_dimension("b", "폭 b")
            h = self._get_dimension("h", "높이 h")
            area = b * h
        elif section == "원형봉":
            d = self._get_dimension("d", "직경 d")
            area = np.pi * d * d / 4
        elif section == "원형관":
            diameter = self._get_dimension("D", "외경 D")
            thickness = self._get_dimension("t", "두께 t")
            if 2 * thickness >= diameter:
                inner_diameter = diameter - 2 * thickness
                raise ValueError(
                    "원형관 내부 공간이 없습니다. 내경 D - 2t는 0보다 커야 합니다. "
                    f"(현재 D={diameter:g} mm, t={thickness:g} mm, "
                    f"내경={inner_diameter:g} mm; t < D/2 조건을 확인하세요.)"
                )
            inner_diameter = diameter - 2 * thickness
            area = np.pi * (diameter * diameter - inner_diameter * inner_diameter) / 4
        elif section == "사각형관":
            width = self._get_dimension("B", "외측 폭 B")
            height = self._get_dimension("H", "외측 높이 H")
            thickness = self._get_dimension("t", "두께 t")
            if 2 * thickness >= min(width, height):
                raise ValueError(
                    "사각형관 내부 공간이 없습니다. 내측 폭 B - 2t와 "
                    "내측 높이 H - 2t가 모두 0보다 커야 합니다. "
                    f"(현재 B={width:g} mm, H={height:g} mm, t={thickness:g} mm)"
                )
            area = width * height - (width - 2 * thickness) * (
                height - 2 * thickness
            )
        else:
            height = self._get_dimension("H", "전체 높이 H")
            width = self._get_dimension("B", "플랜지 폭 B")
            web_thickness = self._get_dimension("tw", "웨브 두께 tw")
            flange_thickness = self._get_dimension("tf", "플랜지 두께 tf")
            if 2 * flange_thickness >= height:
                raise ValueError(
                    "H형강은 플랜지 사이에 웨브가 있어야 합니다. "
                    f"전체 높이 H={height:g} mm는 플랜지 두께 2tf="
                    f"{2 * flange_thickness:g} mm보다 커야 합니다."
                )
            if web_thickness > width:
                raise ValueError("H형강은 웨브 두께 tw가 플랜지 폭 B 이하여야 합니다.")
            area = (
                2 * width * flange_thickness
                + (height - 2 * flange_thickness) * web_thickness
            )

        if not np.isfinite(area) or area <= 0:
            raise ValueError("계산된 단면적이 유효하지 않습니다.")
        return float(area)

    def select_csv(self):
        selected_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=(("CSV 파일", "*.csv"), ("모든 파일", "*.*")),
        )
        if not selected_path:
            return

        self.path.set(selected_path)
        self._invalidate_results("조건변경됨: 다시 분석필요")

    def _invalidate_results(self, status):
        self.area_text.set("단면적 A: - mm²")
        self.valid_count.set("유효 데이터 수: -")
        self.result_text.set("최대 응력: - MPa | 해당 시간: - s")
        self.condition_status.set(status)
        self.input_status.set("")
        self._set_empty_plot()
        self.canvas.draw()

    def analyze(self):
        self._invalidate_results("분석 결과를 갱신하지 못했습니다.")
        if not self.path.get():
            messagebox.showerror("분석 오류", "먼저 CSV 파일을 선택하세요.")
            return

        try:
            area = self._calculate_area()
        except ValueError as error:
            self.condition_status.set("조건변경됨: 다시 분석필요")
            self.input_status.set(str(error))
            messagebox.showerror("치수 입력 오류", str(error))
            return

        try:
            data = pd.read_csv(self.path.get(), dtype=str, keep_default_na=False)
            missing_columns = [
                column
                for column in (TIME_COLUMN, FORCE_COLUMN)
                if column not in data.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"필수 열을 찾을 수 없습니다: {', '.join(missing_columns)}"
                )

            time_values = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
            force_values = pd.to_numeric(data[FORCE_COLUMN], errors="coerce")
            valid_rows = (
                time_values.notna()
                & np.isfinite(time_values)
                & force_values.notna()
                & np.isfinite(force_values)
            )
            valid_time = time_values.loc[valid_rows]
            valid_force = force_values.loc[valid_rows]

            if valid_time.empty:
                self.condition_status.set("분석 실패: 유효한 데이터가 없습니다.")
                messagebox.showinfo("분석 결과", "유효한 데이터가 없습니다.")
                return

            stress = valid_force / area
            max_index = stress.idxmax()
            max_stress = stress.loc[max_index]
            max_time = valid_time.loc[max_index]
            self.area_text.set(f"단면적 A: {area:.6g} mm²")
            self.valid_count.set(f"유효 데이터 수: {len(valid_time)}개")
            self.result_text.set(
                f"최대 응력: {max_stress:g} MPa | 해당 시간: {max_time:g} s"
            )
            self.condition_status.set("")
            self.input_status.set("")

            self._set_empty_plot()
            self.axes.plot(valid_time, stress, marker="o")
            self.axes.set_xlabel("Time(s)")
            self.axes.set_ylabel("Stress(MPa)")
            self.axes.grid(True)
            self.figure.tight_layout()
            self.canvas.draw()
        except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as error:
            self.condition_status.set("분석 실패: 결과가 갱신되지 않았습니다.")
            messagebox.showerror("분석 오류", str(error))


def main():
    root = tk.Tk()
    LoadAnalyzerGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
