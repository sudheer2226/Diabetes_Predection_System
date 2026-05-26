from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv("diabetes_prediction_dataset.csv")

features = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

X = data[features]
y = data["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

try:
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )

except:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""
    probability = ""

    if request.method == "POST":

        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        hba1c = float(request.form["hba1c"])
        glucose = float(request.form["glucose"])

        input_data = pd.DataFrame(
            [[age, bmi, hba1c, glucose]],
            columns=features
        )

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        prob = model.predict_proba(input_scaled)[0][1] * 100

        probability = f"{prob:.2f}%"

        if prediction == 1:
            prediction_text = "Diabetes Detected"
        else:
            prediction_text = "No Diabetes Detected"

    return render_template(
        "index.html",
        prediction=prediction_text,
        probability=probability,
        accuracy=round(accuracy * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)