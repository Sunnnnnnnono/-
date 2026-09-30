import math
import tkinter as tk
from tkinter import messagebox, ttk


FORCE_TO_NEWTON = {
    "N": 1,
    "kN": 1000,
    "kgf": 9.80665,
}
SUPERSCRIPT_TRANSLATION = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def convert_force(value, unit):
    """Convert a force value to N, kN, and kgf."""
    newtons = value * FORCE_TO_NEWTON[unit]
    return newtons, newtons / FORCE_TO_NEWTON["kN"], newtons / FORCE_TO_NEWTON["kgf"]


def format_force(value):
    """Format large force values with scientific notation."""
    if value == 0:
        return "0.00"

    if abs(value) >= 1_000_000:
        exponent = math.floor(math.log10(abs(value)))
        coefficient = value / (10 ** exponent)
        superscript_exponent = str(exponent).translate(SUPERSCRIPT_TRANSLATION)
        return f"{coefficient:.2f} x 10{superscript_exponent}"

    return f"{value:,.2f}"


def main():
    root = tk.Tk()
    root.title("힘 단위 변환기")
    root.geometry("430x600")
    root.resizable(False, False)

    history = []
    value_var = tk.StringVar()
    unit_var = tk.StringVar(value="kN")
    result_var = tk.StringVar(value="값과 단위를 입력한 뒤 변환 버튼을 누르세요.")

    def show_invalid_input_warning():
        messagebox.showwarning("입력 오류", "숫자만 입력 가능합니다.", parent=root)
        value_entry.focus_set()

    def convert_from_input():
        user_input = value_var.get().strip()
        unit = unit_var.get()

        if not user_input:
            show_invalid_input_warning()
            return

        try:
            value = float(user_input)
        except ValueError:
            show_invalid_input_warning()
            return

        if not math.isfinite(value):
            show_invalid_input_warning()
            return

        newtons, kilonewtons, kilogram_force = convert_force(value, unit)
        result_var.set(
            f"N: {format_force(newtons)} N\n"
            f"kN: {format_force(kilonewtons)} kN\n"
            f"kgf: {format_force(kilogram_force)} kgf"
        )
        history.append(f"{format_force(value)} {unit} -> {format_force(newtons)} N, "
                       f"{format_force(kilonewtons)} kN, {format_force(kilogram_force)} kgf")

    def clear_input():
        value_var.set("")
        result_var.set("값과 단위를 입력한 뒤 변환 버튼을 누르세요.")
        value_entry.focus_set()

    def append_to_input(value):
        current_value = value_var.get()
        if value == "." and "." in current_value:
            return
        value_var.set(current_value + value)
        value_entry.focus_set()

    def backspace_input():
        value_var.set(value_var.get()[:-1])
        value_entry.focus_set()

    def toggle_sign():
        current_value = value_var.get()
        if current_value.startswith("-"):
            value_var.set(current_value[1:])
        elif current_value:
            value_var.set("-" + current_value)
        value_entry.focus_set()

    def show_history():
        history_window = tk.Toplevel(root)
        history_window.title("변환 기록")
        history_window.geometry("650x260")
        history_window.resizable(False, False)

        history_list = tk.Listbox(history_window, width=92, height=11)
        history_list.pack(padx=12, pady=12, fill="both", expand=True)
        if history:
            for item in history:
                history_list.insert(tk.END, item)
        else:
            history_list.insert(tk.END, "아직 변환 기록이 없습니다.")

    content = ttk.Frame(root, padding=20)
    content.pack(fill="both", expand=True)
    ttk.Label(content, text="힘 단위 변환기", font=("맑은 고딕", 16, "bold")).pack(pady=(0, 18))

    input_frame = ttk.Frame(content)
    input_frame.pack(fill="x")
    ttk.Label(input_frame, text="값").grid(row=0, column=0, padx=(0, 8), pady=6)
    value_entry = ttk.Entry(input_frame, textvariable=value_var, width=22)
    value_entry.grid(row=0, column=1, padx=(0, 8), pady=6)
    ttk.Combobox(input_frame, textvariable=unit_var,
                 values=tuple(FORCE_TO_NEWTON), state="readonly", width=8).grid(
                     row=0, column=2, pady=6)

    keypad_frame = ttk.LabelFrame(content, text="숫자 입력", padding=8)
    keypad_frame.pack(pady=(4, 8))
    keypad_buttons = [
        ("7", lambda: append_to_input("7")),
        ("8", lambda: append_to_input("8")),
        ("9", lambda: append_to_input("9")),
        ("⌫", backspace_input),
        ("4", lambda: append_to_input("4")),
        ("5", lambda: append_to_input("5")),
        ("6", lambda: append_to_input("6")),
        ("C", clear_input),
        ("1", lambda: append_to_input("1")),
        ("2", lambda: append_to_input("2")),
        ("3", lambda: append_to_input("3")),
        ("±", toggle_sign),
        ("0", lambda: append_to_input("0")),
        (".", lambda: append_to_input(".")),
    ]
    for index, (label, command) in enumerate(keypad_buttons):
        ttk.Button(keypad_frame, text=label, command=command, width=6).grid(
            row=index // 4, column=index % 4, padx=3, pady=3, ipady=5)

    button_frame = ttk.Frame(content)
    button_frame.pack(pady=12)
    ttk.Button(button_frame, text="변환", command=convert_from_input).grid(row=0, column=0, padx=4)
    ttk.Button(button_frame, text="입력 지우기", command=clear_input).grid(row=0, column=1, padx=4)
    ttk.Button(button_frame, text="변환 기록 보기", command=show_history).grid(row=0, column=2, padx=4)

    result_frame = ttk.LabelFrame(content, text="계산 결과", padding=12)
    result_frame.pack(fill="both", expand=True, pady=(4, 0))
    ttk.Label(result_frame, textvariable=result_var, justify="left", anchor="w").pack(fill="both", expand=True)

    value_entry.bind("<Return>", lambda event: convert_from_input())
    value_entry.focus_set()
    root.mainloop()


if __name__ == "__main__":
    main()