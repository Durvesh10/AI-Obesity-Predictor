import os
import gradio as gr
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

obesity = fetch_ucirepo(id=544)

X_raw = obesity.data.features
y = obesity.data.targets.iloc[:, 0]

df = X_raw.copy()
df["NObeyesdad"] = y


# ============================================================
# 2. MODEL FEATURES
# ============================================================

# Height and Weight are kept for BMI calculation only.
X = df.drop(columns=["NObeyesdad", "Height", "Weight"])
y = df["NObeyesdad"]


categorical_features = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]

numerical_features = [
    "Age",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]


# ============================================================
# 3. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 4. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)


# ============================================================
# 6. LABELS
# ============================================================

friendly_labels = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III"
}


# ============================================================
# 7. BMI FUNCTIONS
# ============================================================

def calculate_bmi(height, weight):

    if height is None or weight is None:
        return 0

    if height <= 0 or weight <= 0:
        return 0

    return weight / (height ** 2)


def get_bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal"

    elif bmi < 30:
        return "Overweight"

    else:
        return "Obesity"


# ============================================================
# 8. SIMPLE BMI DISPLAY
# ============================================================

def create_bmi_html(bmi):

    category = get_bmi_category(bmi)

    if category == "Underweight":
        position = 15
        category_color = "#2563eb"

    elif category == "Normal":
        position = 40
        category_color = "#16a34a"

    elif category == "Overweight":
        position = 65
        category_color = "#ca8a04"

    else:
        position = 88
        category_color = "#dc2626"

    return f"""
    <div style="
        border:1px solid #d1d5db;
        border-radius:8px;
        padding:18px;
        background:white;
        margin-top:10px;
    ">

        <div style="
            font-size:28px;
            font-weight:bold;
            color:#111827;
        ">
            BMI: {bmi:.1f}
        </div>

        <div style="
            margin-top:5px;
            font-size:18px;
            font-weight:bold;
            color:{category_color};
        ">
            {category}
        </div>

        <div style="
            position:relative;
            margin-top:20px;
            height:22px;
            border-radius:5px;
            overflow:hidden;
            background:linear-gradient(
                to right,
                #93c5fd 0%,
                #93c5fd 25%,
                #86efac 25%,
                #86efac 50%,
                #fde047 50%,
                #fde047 75%,
                #fca5a5 75%,
                #fca5a5 100%
            );
        ">

            <div style="
                position:absolute;
                left:{position}%;
                top:-5px;
                width:4px;
                height:32px;
                background:#111827;
                border-radius:2px;
            ">
            </div>

        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            font-size:12px;
            color:#374151;
            margin-top:7px;
        ">
            <span>Underweight</span>
            <span>Normal</span>
            <span>Overweight</span>
            <span>Obesity</span>
        </div>

    </div>
    """


# ============================================================
# 9. PROBABILITY DISPLAY
# ============================================================

def create_probability_html(probabilities, classes):

    rows = ""

    probability_data = []

    for cls, probability in zip(classes, probabilities):

        label = friendly_labels.get(cls, cls)

        probability_data.append(
            (label, probability * 100)
        )

    probability_data.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for label, probability in probability_data:

        rows += f"""
        <div style="margin-bottom:12px;">

            <div style="
                display:flex;
                justify-content:space-between;
                margin-bottom:4px;
                font-size:14px;
                color:#111827;
            ">

                <span>{label}</span>

                <strong>{probability:.1f}%</strong>

            </div>

            <div style="
                width:100%;
                height:10px;
                background:#e5e7eb;
                border-radius:5px;
                overflow:hidden;
            ">

                <div style="
                    width:{probability}%;
                    height:100%;
                    background:#2563eb;
                    border-radius:5px;
                ">
                </div>

            </div>

        </div>
        """

    return f"""
    <div style="
        border:1px solid #d1d5db;
        border-radius:8px;
        padding:18px;
        background:white;
        margin-top:10px;
    ">

        {rows}

    </div>
    """


# ============================================================
# 10. HABIT COACH
# ============================================================

def generate_habit_coach(
    fcvc,
    ncp,
    ch2o,
    faf,
    tue,
    favc,
    caec,
    smoke,
    calc
):

    suggestions = []

    # Vegetable intake
    if fcvc < 2:
        suggestions.append(
            "Increase vegetable and fruit intake."
        )

    # Meals
    if ncp < 3:
        suggestions.append(
            "Try to maintain regular and balanced meals."
        )

    # Water
    if ch2o < 2:
        suggestions.append(
            "Increase daily water intake."
        )

    # Physical activity
    if faf < 2:
        suggestions.append(
            "Increase regular physical activity such as walking, cycling or exercise."
        )

    # Technology usage
    if tue > 2:
        suggestions.append(
            "Reduce unnecessary screen and sedentary time."
        )

    # High calorie food
    if favc == "yes":
        suggestions.append(
            "Reduce frequent consumption of high-calorie foods."
        )

    # Eating between meals
    if caec == "Always":
        suggestions.append(
            "Reduce frequent snacking between meals."
        )

    # Smoking
    if smoke == "yes":
        suggestions.append(
            "Consider reducing or avoiding smoking."
        )

    # Alcohol
    if calc in ["Frequently", "Always"]:
        suggestions.append(
            "Reduce frequent alcohol consumption."
        )

    if not suggestions:
        suggestions.append(
            "Your current habits look relatively balanced. "
            "Continue maintaining healthy eating and activity habits."
        )

    result = "### Habit Suggestions\n\n"

    for suggestion in suggestions:
        result += f"- {suggestion}\n"

    return result


# ============================================================
# 11. ACTION PLAN
# ============================================================

def generate_action_plan(
    fcvc,
    ch2o,
    faf,
    tue,
    favc
):

    output = ""

    output += "**1. Physical Activity**\n\n"

    if faf < 2:
        output += (
            "Aim for more regular physical activity. "
            "Start with walking or light exercise and gradually increase duration.\n\n"
        )
    else:
        output += (
            "Continue your current physical activity and maintain a regular routine.\n\n"
        )

    output += "**2. Food Habits**\n\n"

    if favc == "yes":
        output += (
            "Reduce high-calorie and highly processed foods. "
            "Prefer balanced meals with vegetables, fruits and protein.\n\n"
        )
    else:
        output += (
            "Continue maintaining balanced food choices and regular meals.\n\n"
        )

    output += "**3. Water Intake**\n\n"

    if ch2o < 2:
        output += (
            "Try to increase your daily water intake gradually.\n\n"
        )
    else:
        output += (
            "Maintain adequate daily water intake.\n\n"
        )

    output += "**4. Daily Routine**\n\n"

    if tue > 2:
        output += (
            "Reduce prolonged screen time and include short movement breaks "
            "during long periods of sitting.\n\n"
        )
    else:
        output += (
            "Maintain an active daily routine and avoid prolonged inactivity.\n\n"
        )

    output += (
        "**Note:** These suggestions are general lifestyle guidance "
        "and are not a medical treatment plan."
    )

    return output


# ============================================================
# 12. PREDICTION FUNCTION
# ============================================================

def predict_obesity(
    age,
    gender,
    height,
    weight,
    family_history,
    favc,
    fcvc,
    ncp,
    caec,
    smoke,
    ch2o,
    scc,
    faf,
    tue,
    calc,
    mtrans
):

    # -----------------------------
    # Validate input
    # -----------------------------

    if height is None or weight is None:
        return (
            "Please enter height and weight.",
            "",
            "",
            ""
        )

    if height <= 0 or weight <= 0:
        return (
            "Please enter valid height and weight values.",
            "",
            "",
            ""
        )

    # -----------------------------
    # BMI
    # -----------------------------

    bmi = calculate_bmi(height, weight)

    bmi_html = create_bmi_html(bmi)

    # -----------------------------
    # Prepare model input
    # -----------------------------

    user_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = pipeline.predict(user_data)[0]

    probabilities = pipeline.predict_proba(user_data)[0]

    classes = pipeline.classes_

    predicted_label = friendly_labels.get(
        prediction,
        prediction
    )

    # -----------------------------
    # Estimated unhealthy risk
    # -----------------------------

    healthy_classes = [
        "Insufficient_Weight",
        "Normal_Weight"
    ]

    unhealthy_probability = 0

    for cls, probability in zip(
        classes,
        probabilities
    ):

        if cls not in healthy_classes:
            unhealthy_probability += probability

    risk_percentage = unhealthy_probability * 100

    # -----------------------------
    # Prediction result
    # -----------------------------

    result_html = f"""
    <div style="
        border:1px solid #d1d5db;
        border-radius:8px;
        padding:20px;
        background:white;
        margin-top:10px;
    ">

        <div style="
            font-size:15px;
            color:#4b5563;
        ">
            Predicted Obesity Level
        </div>

        <div style="
            font-size:30px;
            font-weight:bold;
            color:#2563eb;
            margin-top:5px;
        ">
            {predicted_label}
        </div>

        <hr style="
            border:none;
            border-top:1px solid #e5e7eb;
            margin:15px 0;
        ">

        <div style="
            font-size:15px;
            color:#4b5563;
        ">
            Estimated model probability of an unhealthy category
        </div>

        <div style="
            font-size:25px;
            font-weight:bold;
            color:#111827;
            margin-top:5px;
        ">
            {risk_percentage:.1f}%
        </div>

    </div>
    """

    # -----------------------------
    # Probability chart
    # -----------------------------

    probability_html = create_probability_html(
        probabilities,
        classes
    )

    # -----------------------------
    # Habit Coach
    # -----------------------------

    habit_coach = generate_habit_coach(
        fcvc,
        ncp,
        ch2o,
        faf,
        tue,
        favc,
        caec,
        smoke,
        calc
    )

    # -----------------------------
    # Action Plan
    # -----------------------------

    action_plan = generate_action_plan(
        fcvc,
        ch2o,
        faf,
        tue,
        favc
    )

    return (
        result_html,
        bmi_html,
        probability_html,
        habit_coach,
        action_plan
    )


# ============================================================
# 13. SIMPLE CSS
# ============================================================

css = """

body {
    background: #f8fafc !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}

h1 {
    color: #111827 !important;
}

h2, h3 {
    color: #1f2937 !important;
}

label {
    color: #111827 !important;
}

button.primary {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 12px 25px !important;
}

button.primary:hover {
    background: #1d4ed8 !important;
}

.info-box {
    background: white !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 15px 18px !important;
    margin-bottom: 15px !important;
}

.info-box,
.info-box p,
.info-box li,
.info-box span,
.info-box div,
.info-box strong,
.info-box b {
    color: #111827 !important;
}

.section-title {
    border-bottom: 1px solid #d1d5db;
    padding-bottom: 8px;
    margin-top: 20px;
}

.disclaimer {
    background: #fefce8 !important;
    color: #713f12 !important;
    border: 1px solid #fde68a !important;
    border-radius: 8px !important;
    padding: 15px !important;
}

"""


# ============================================================
# 14. GRADIO UI
# ============================================================

with gr.Blocks(
    title="AI/ML Obesity Level Predictor",
    css=css
) as app:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        # AI/ML Obesity Level Predictor

        ### Personalized Habit Coaching System

        This application predicts obesity level using physical,
        behavioral and lifestyle information and provides simple
        habit-based recommendations.
        """
    )

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    gr.Markdown("## About Application")

    gr.HTML(
        """
        <div class="info-box">

        <p>
        This project uses Machine Learning to predict obesity levels
        from lifestyle and demographic information.
        </p>

        <p>
        A Random Forest classification model is trained using the
        UCI Obesity Dataset. The application also calculates BMI
        separately and provides simple lifestyle suggestions.
        </p>

        </div>
        """
    )

    gr.Markdown("## How to Enter Values")

    gr.HTML(
        """
        <div class="info-box">

        <ul>
            <li>Enter your information in the fields below.</li>
            <li>Height should be entered in metres.</li>
            <li>Weight should be entered in kilograms.</li>
            <li>Use the values that best describe your usual habits.</li>
            <li>Click <b>Predict Obesity Level</b> to see the results.</li>
        </ul>

        </div>
        """
    )

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    gr.Markdown("## Personal Information")

    with gr.Row():

        age = gr.Number(
            label="Age",
            value=22,
            minimum=1,
            maximum=100
        )

        gender = gr.Dropdown(
            choices=["Male", "Female"],
            label="Gender",
            value="Male"
        )

    with gr.Row():

        height = gr.Number(
            label="Height (metres)",
            value=1.70,
            minimum=0.5,
            maximum=2.5
        )

        weight = gr.Number(
            label="Weight (kg)",
            value=65,
            minimum=10,
            maximum=300
        )

    # --------------------------------------------------------
    # FAMILY & EATING HABITS
    # --------------------------------------------------------

    gr.Markdown("## Family & Eating Habits")

    with gr.Row():

        family_history = gr.Dropdown(
            choices=["yes", "no"],
            label="Family History of Overweight",
            value="no"
        )

        favc = gr.Dropdown(
            choices=["yes", "no"],
            label="Frequent High-Calorie Food",
            value="no"
        )

    with gr.Row():

        fcvc = gr.Slider(
            minimum=1,
            maximum=3,
            step=1,
            value=2,
            label="Vegetable Intake"
        )

        ncp = gr.Slider(
            minimum=1,
            maximum=4,
            step=1,
            value=3,
            label="Number of Main Meals"
        )

    with gr.Row():

        caec = gr.Dropdown(
            choices=[
                "no",
                "Sometimes",
                "Frequently",
                "Always"
            ],
            label="Eating Between Meals",
            value="Sometimes"
        )

        ch2o = gr.Slider(
            minimum=1,
            maximum=3,
            step=0.1,
            value=2,
            label="Daily Water Intake"
        )

    # --------------------------------------------------------
    # PHYSICAL ACTIVITY & OTHER HABITS
    # --------------------------------------------------------

    gr.Markdown("## Physical Activity & Other Habits")

    with gr.Row():

        smoke = gr.Dropdown(
            choices=["yes", "no"],
            label="Smoking",
            value="no"
        )

        scc = gr.Dropdown(
            choices=["yes", "no"],
            label="Calories Monitoring",
            value="no"
        )

    with gr.Row():

        faf = gr.Slider(
            minimum=0,
            maximum=3,
            step=1,
            value=1,
            label="Physical Activity"
        )

        tue = gr.Slider(
            minimum=0,
            maximum=2,
            step=0.1,
            value=1,
            label="Technology / Screen Usage"
        )

    with gr.Row():

        calc = gr.Dropdown(
            choices=[
                "no",
                "Sometimes",
                "Frequently",
                "Always"
            ],
            label="Alcohol Consumption",
            value="Sometimes"
        )

        mtrans = gr.Dropdown(
            choices=[
                "Automobile",
                "Motorbike",
                "Bike",
                "Public_Transportation",
                "Walking"
            ],
            label="Main Transportation",
            value="Public_Transportation"
        )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    gr.Markdown("")

    predict_button = gr.Button(
        "Predict Obesity Level",
        variant="primary"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    gr.Markdown("## Prediction Results")

    result_output = gr.HTML()

    gr.Markdown("### BMI Indicator")

    bmi_output = gr.HTML()

    gr.Markdown("### Prediction Probability")

    probability_output = gr.HTML()

    # --------------------------------------------------------
    # UNDERSTANDING RESULTS
    # --------------------------------------------------------

    gr.Markdown("## Understanding Results")

    gr.HTML(
        """
        <div class="info-box">

        <p>
        The predicted obesity level is generated by the Random Forest
        machine-learning model using the information entered above.
        </p>

        <p>
        BMI is calculated separately using height and weight:
        </p>

        <p>
        <b>BMI = Weight (kg) / Height² (m²)</b>
        </p>

        <p>
        The probability values show how strongly the trained model
        associates the input with each obesity category.
        </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # HABIT COACH
    # --------------------------------------------------------

    gr.Markdown("## Simple Habit Coach")

    habit_output = gr.Markdown(
        "Your habit suggestions will appear here."
    )

    # --------------------------------------------------------
    # ACTION PLAN
    # --------------------------------------------------------

    gr.Markdown("## Personalized Action Plan")

    action_output = gr.Markdown(
        "Your action plan will appear here."
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    gr.Markdown("## About AI/ML Model")

    gr.HTML(
        """
        <div class="info-box">

        <ul>
            <li><b>Dataset:</b> UCI Obesity Dataset</li>
            <li><b>Dataset ID:</b> 544</li>
            <li><b>Model:</b> Random Forest Classifier</li>
            <li><b>Number of Trees:</b> 300</li>
            <li><b>Train-Test Split:</b> 80:20</li>
            <li><b>Classification:</b> 7 obesity categories</li>
            <li><b>Framework:</b> Scikit-learn</li>
            <li><b>Interface:</b> Gradio</li>
        </ul>

        </div>
        """
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    gr.Markdown("## Disclaimer")

    gr.HTML(
        """
        <div class="disclaimer">

        <b>Important:</b>

        <p>
        This application is developed for educational and
        demonstration purposes only. The predictions and
        recommendations are not medical diagnoses and should
        not replace professional medical advice.
        </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # BUTTON FUNCTION
    # --------------------------------------------------------

    predict_button.click(
        fn=predict_obesity,
        inputs=[
            age,
            gender,
            height,
            weight,
            family_history,
            favc,
            fcvc,
            ncp,
            caec,
            smoke,
            ch2o,
            scc,
            faf,
            tue,
            calc,
            mtrans
        ],
        outputs=[
            result_output,
            bmi_output,
            probability_output,
            habit_output,
            action_output
        ]
    )


# ============================================================
# 15. LAUNCH
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
