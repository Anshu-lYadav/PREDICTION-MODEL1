import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load data and model
try:
    teams = pd.read_csv("teams.csv")
except FileNotFoundError:
    print("Error: 'teams.csv' not found.")
    exit()

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
teams['prev_medals'] = teams['prev_medals'].fillna(0)

features = ['athletes', 'age', 'prev_medals']
target = 'medals'

X = teams[features]
y = teams[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# GUI setup
window = tk.Tk()
window.title("🎖 Olympic Medals Predictor")
window.geometry("850x800")
window.configure(bg="#f0f0f0")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#f0f0f0", foreground="#333", font=("Segoe UI", 11))
style.configure("TEntry", font=("Segoe UI", 11))
style.configure("TButton", font=("Segoe UI", 11, "bold"), background="#3333cc", foreground="#ffffff")
style.configure("TCombobox", font=("Segoe UI", 11))

header = tk.Label(window, text="🏅 Olympic Medals Predictor", font=("Segoe UI", 20, "bold"), bg="#f0f0f0", fg="#3333cc")
header.pack(pady=10)

input_frame = ttk.LabelFrame(window, text="🎯 Enter Team Details")
input_frame.pack(padx=20, pady=10, fill="x")

# Form layout
def add_row(label_text, row, column, is_combo=False, values=[]):
    label = ttk.Label(input_frame, text=label_text)
    label.grid(row=row, column=column*2, padx=10, pady=8, sticky="e")
    if is_combo:
        box = ttk.Combobox(input_frame, values=values, state="readonly", width=22)
        box.grid(row=row, column=column*2+1, padx=10, pady=8)
        return box
    else:
        entry = ttk.Entry(input_frame, width=25)
        entry.grid(row=row, column=column*2+1, padx=10, pady=8)
        return entry

athletes_entry = add_row("Number of Athletes:", 0, 0)
age_entry = add_row("Average Age:", 1, 0)
prev_medals_entry = add_row("Previous Medals:", 2, 0)

country_combobox = add_row("Country:", 0, 1, is_combo=True, values=sorted(teams['country'].unique()))
year_combobox = add_row("Year:", 1, 1, is_combo=True, values=sorted(set(teams['year'].unique()).union({2020, 2024}), reverse=True))

# Frames for results and charts
predict_button = ttk.Button(window, text="🔮 Predict Medals", command=lambda: [predict_medals(), draw_line_graph()])
predict_button.pack(pady=15)

result_frame = ttk.LabelFrame(window, text="📢 Prediction Result")
result_frame.pack(padx=20, pady=5, fill="x")
result_label = ttk.Label(result_frame, text="Fill in the details and click Predict", font=("Segoe UI", 11, "italic"))
result_label.pack(pady=10)

evaluation_frame = ttk.LabelFrame(window, text="📈 Model Evaluation (Test Data)")
evaluation_frame.pack(padx=20, pady=5, fill="x")
mae_label = ttk.Label(evaluation_frame, text=f"📉 Mean Absolute Error: {mae:.2f}")
mae_label.pack(padx=10, pady=5)
r2_label = ttk.Label(evaluation_frame, text=f"📊 R-squared Score: {r2:.2f}")
r2_label.pack(padx=10, pady=5)

chart_frame = ttk.LabelFrame(window, text="🎨 Visual Prediction & History")
chart_frame.pack(padx=20, pady=10, fill="both", expand=True)
pie_canvas = None
line_canvas = None

# Prediction function
# Prediction function
def predict_medals():
    try:
        athletes = int(athletes_entry.get())
        age = float(age_entry.get())
        prev_medals = float(prev_medals_entry.get())

        new_data = pd.DataFrame({'athletes': [athletes], 'age': [age], 'prev_medals': [prev_medals]})
        predicted_medals = model.predict(new_data)[0]

        # Clamp prediction between 0 and 100
        predicted_medals = min(max(0, predicted_medals), 100)

        selected_year = year_combobox.get()
        if selected_year:
            next_olympics_year = int(selected_year) + 4
            selected_year_text = f" for {next_olympics_year} (based on {selected_year})"
        else:
            selected_year_text = ""

        result_label.config(text=f"🏅 Predicted Medals{selected_year_text}: {predicted_medals:.2f}")
        draw_pie_chart(predicted_medals)

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")

# Autofill function
def autofill_fields(event=None):
    selected_country = country_combobox.get()
    selected_year = year_combobox.get()

    country_data = teams[teams['country'] == selected_country]
    if selected_year:
        country_data = country_data[country_data['year'] == int(selected_year)]

    country_data = country_data.sort_values(by="year", ascending=False)

    if not country_data.empty:
        latest = country_data.iloc[0]
        athletes_entry.delete(0, tk.END)
        age_entry.delete(0, tk.END)
        prev_medals_entry.delete(0, tk.END)

        athletes_entry.insert(0, str(int(latest['athletes'])))
        age_entry.insert(0, str(float(latest['age'])))
        prev_medals_entry.insert(0, str(float(latest['prev_medals'])))
        draw_line_graph()

# Pie chart
def draw_pie_chart(predicted):
    global pie_canvas
    for widget in chart_frame.winfo_children():
        widget.destroy()

    max_medals = 100
    predicted = max(0, min(predicted, max_medals))

    labels = ['Predicted Medals', 'Remaining']
    values = [predicted, max_medals - predicted]
    colors = ['#3CB371', '#FF6347']

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    pie_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    pie_canvas.draw()
    pie_canvas.get_tk_widget().pack(side=tk.LEFT, fill="both", expand=True)

# Line graph
def draw_line_graph():
    global line_canvas
    selected_country = country_combobox.get()
    if not selected_country:
        return

    history = teams[teams['country'] == selected_country].sort_values("year")
    if history.empty:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(history["year"], history["medals"], marker='o', color="#1f77b4")
    ax.set_title(f"{selected_country} Medal History")
    ax.set_xlabel("Year")
    ax.set_ylabel("Medals")
    ax.grid(True)

    line_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    line_canvas.draw()
    line_canvas.get_tk_widget().pack(side=tk.RIGHT, fill="both", expand=True)

# Bind autofill
country_combobox.bind("<<ComboboxSelected>>", autofill_fields)
year_combobox.bind("<<ComboboxSelected>>", autofill_fields)

window.mainloop()
