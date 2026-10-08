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

# Height and Weight are used separately for BMI calculation.
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
# 6. FRIENDLY LABELS
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
# 8. BMI DISPLAY
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
    <div class="bmi-card">

        <div class="bmi-number">
            BMI: {bmi:.1f}
        </div>

        <div class="bmi-category" style="color:{category_color};">
            {category}
        </div>

        <div class="bmi-scale">

            <div class="bmi-marker"
                 style="left:{position}%;">
            </div>

        </div>

        <div class="bmi-labels">
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

    rows = ""

    for label, probability in probability_data:

        rows += f"""
        <div class="probability-row">

            <div class="probability-header">

                <span>{label}</span>

                <strong>{probability:.1f}%</strong>

            </div>

            <div class="probability-track">

                <div class="probability-fill"
                     style="width:{probability}%;">
                </div>

            </div>

        </div>
        """

    return f"""
    <div class="probability-card">

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

    if fcvc < 2:
        suggestions.append(
            "Increase vegetable and fruit intake."
        )

    if ncp < 3:
        suggestions.append(
            "Try to maintain regular and balanced meals."
        )

    if ch2o < 2:
        suggestions.append(
            "Increase daily water intake."
        )

    if faf < 2:
        suggestions.append(
            "Increase regular physical activity such as walking, cycling or exercise."
        )

    if tue > 2:
        suggestions.append(
            "Reduce unnecessary screen and sedentary time."
        )

    if favc == "yes":
        suggestions.append(
            "Reduce frequent consumption of high-calorie foods."
        )

    if caec == "Always":
        suggestions.append(
            "Reduce frequent snacking between meals."
        )

    if smoke == "yes":
        suggestions.append(
            "Consider reducing or avoiding smoking."
        )

    if calc in ["Frequently", "Always"]:
        suggestions.append(
            "Reduce frequent alcohol consumption."
        )

    if not suggestions:
        suggestions.append(
            "Your current habits look relatively balanced. "
            "Continue maintaining healthy eating and activity habits."
        )

    result = ""

    for suggestion in suggestions:
        result += f"• {suggestion}\n\n"

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

    if height is None or weight is None:

        return (
            """
            <div class="error-box">
                Please enter height and weight.
            </div>
            """,
            "",
            "",
            "",
            ""
        )

    if height <= 0 or weight <= 0:

        return (
            """
            <div class="error-box">
                Please enter valid height and weight values.
            </div>
            """,
            "",
            "",
            "",
            ""
        )

    # --------------------------------------------------------
    # BMI
    # --------------------------------------------------------

    bmi = calculate_bmi(height, weight)

    bmi_html = create_bmi_html(bmi)

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = pipeline.predict(user_data)[0]

    probabilities = pipeline.predict_proba(user_data)[0]

    classes = pipeline.classes_

    predicted_label = friendly_labels.get(
        prediction,
        prediction
    )

    # --------------------------------------------------------
    # UNHEALTHY CATEGORY RISK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_html = f"""
    <div class="result-card">

        <div class="result-label">
            Predicted Obesity Level
        </div>

        <div class="prediction-value">
            {predicted_label}
        </div>

        <div class="result-divider"></div>

        <div class="result-label">
            Estimated model probability of an unhealthy category
        </div>

        <div class="risk-value">
            {risk_percentage:.1f}%
        </div>

    </div>
    """

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability_html = create_probability_html(
        probabilities,
        classes
    )

    # --------------------------------------------------------
    # HABIT COACH
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ACTION PLAN
    # --------------------------------------------------------

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
# 13. PROFESSIONAL UI CSS
# ============================================================

css = """

/* ---------------------------------------------------------
   MAIN PAGE
--------------------------------------------------------- */

body {
    background: #f1f5f9 !important;
}

.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
    padding-bottom: 40px !important;
}


/* ---------------------------------------------------------
   MAIN TEXT
--------------------------------------------------------- */

h1 {
    color: #0f172a !important;
    font-size: 34px !important;
    font-weight: 700 !important;
}

h2 {
    color: #0f3d56 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-top: 25px !important;
}

h3 {
    color: #155e75 !important;
    font-size: 19px !important;
}


/* ---------------------------------------------------------
   HEADER
--------------------------------------------------------- */

.header-box {
    background: linear-gradient(
        135deg,
        #0f4c81,
        #087f8c
    ) !important;

    color: white !important;

    border-radius: 14px !important;

    padding: 28px !important;

    margin-bottom: 20px !important;

    box-shadow:
        0 5px 15px rgba(15, 76, 129, 0.15);
}

.header-box h1 {
    color: white !important;
    margin-bottom: 8px !important;
}

.header-box p {
    color: #e0f2fe !important;
}


/* ---------------------------------------------------------
   INFORMATION BOXES
--------------------------------------------------------- */

.info-box {
    background: #ffffff !important;

    color: #1e293b !important;

    border: 1px solid #dbe4ea !important;

    border-left: 5px solid #0891b2 !important;

    border-radius: 10px !important;

    padding: 18px 20px !important;

    margin-bottom: 15px !important;

    box-shadow:
        0 2px 8px rgba(15, 23, 42, 0.05);
}

.info-box,
.info-box p,
.info-box li,
.info-box span,
.info-box div,
.info-box strong,
.info-box b {
    color: #1e293b !important;
}

.info-box ul {
    margin-bottom: 0 !important;
}


/* ---------------------------------------------------------
   INPUT AREAS
--------------------------------------------------------- */

label {
    color: #1e293b !important;
    font-weight: 600 !important;
}

input,
textarea,
select {
    border-radius: 8px !important;
}


/* ---------------------------------------------------------
   PREDICT BUTTON
--------------------------------------------------------- */

button.primary {
    background: linear-gradient(
        135deg,
        #0f4c81,
        #0891b2
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 9px !important;

    font-size: 18px !important;

    font-weight: 700 !important;

    padding: 13px 28px !important;

    box-shadow:
        0 4px 10px rgba(8, 145, 178, 0.25);

    transition: 0.2s;
}

button.primary:hover {
    transform: translateY(-1px);

    box-shadow:
        0 6px 15px rgba(8, 145, 178, 0.35);
}


/* ---------------------------------------------------------
   RESULT CARD
--------------------------------------------------------- */

.result-card {
    background: linear-gradient(
        135deg,
        #eff6ff,
        #ecfeff
    );

    border: 1px solid #bae6fd;

    border-radius: 12px;

    padding: 22px;

    margin-top: 8px;

    box-shadow:
        0 3px 10px rgba(15, 23, 42, 0.06);
}

.result-label {
    color: #475569;

    font-size: 15px;

    font-weight: 600;
}

.prediction-value {
    color: #0f4c81;

    font-size: 30px;

    font-weight: 800;

    margin-top: 6px;
}

.risk-value {
    color: #0f172a;

    font-size: 27px;

    font-weight: 800;

    margin-top: 5px;
}

.result-divider {
    height: 1px;

    background: #cbd5e1;

    margin: 18px 0;
}


/* ---------------------------------------------------------
   BMI CARD
--------------------------------------------------------- */

.bmi-card {
    background: #ffffff;

    border: 1px solid #dbe4ea;

    border-radius: 12px;

    padding: 20px;

    box-shadow:
        0 3px 10px rgba(15, 23, 42, 0.06);
}

.bmi-number {
    color: #0f172a;

    font-size: 30px;

    font-weight: 800;
}

.bmi-category {
    font-size: 19px;

    font-weight: 700;

    margin-top: 4px;
}

.bmi-scale {
    position: relative;

    height: 20px;

    margin-top: 20px;

    border-radius: 10px;

    background: linear-gradient(
        to right,
        #60a5fa 0%,
        #60a5fa 25%,
        #4ade80 25%,
        #4ade80 50%,
        #facc15 50%,
        #facc15 75%,
        #f87171 75%,
        #f87171 100%
    );
}

.bmi-marker {
    position: absolute;

    top: -6px;

    width: 5px;

    height: 32px;

    background: #111827;

    border-radius: 4px;

    box-shadow:
        0 0 0 2px white;
}

.bmi-labels {
    display: flex;

    justify-content: space-between;

    margin-top: 8px;

    font-size: 12px;

    color: #475569;

    font-weight: 600;
}


/* ---------------------------------------------------------
   PROBABILITY CARD
--------------------------------------------------------- */

.probability-card {
    background: #ffffff;

    border: 1px solid #dbe4ea;

    border-radius: 12px;

    padding: 20px;

    margin-top: 5px;

    box-shadow:
        0 3px 10px rgba(15, 23, 42, 0.06);
}

.probability-row {
    margin-bottom: 14px;
}

.probability-header {
    display: flex;

    justify-content: space-between;

    color: #334155;

    font-size: 14px;

    margin-bottom: 5px;
}

.probability-header strong {
    color: #0f4c81;
}

.probability-track {
    width: 100%;

    height: 10px;

    background: #e2e8f0;

    border-radius: 8px;

    overflow: hidden;
}

.probability-fill {
    height: 100%;

    background: linear-gradient(
        90deg,
        #0f4c81,
        #0891b2
    );

    border-radius: 8px;
}


/* ---------------------------------------------------------
   HABIT COACH
--------------------------------------------------------- */

.habit-box {
    background: #f0fdfa;

    border: 1px solid #99f6e4;

    border-left: 5px solid #0d9488;

    border-radius: 10px;

    padding: 18px;

    color: #134e4a;
}


/* ---------------------------------------------------------
   ACTION PLAN
--------------------------------------------------------- */

.action-box {
    background: #eff6ff;

    border: 1px solid #bfdbfe;

    border-left: 5px solid #2563eb;

    border-radius: 10px;

    padding: 18px;

    color: #1e3a8a;
}

.action-box,
.action-box p,
.action-box strong,
.action-box b {
    color: #1e3a8a !important;
}


/* ---------------------------------------------------------
   DISCLAIMER
--------------------------------------------------------- */

.disclaimer {
    background: #fffbeb !important;

    color: #78350f !important;

    border: 1px solid #fde68a !important;

    border-left: 5px solid #f59e0b !important;

    border-radius: 10px !important;

    padding: 18px !important;
}

.disclaimer,
.disclaimer p,
.disclaimer strong,
.disclaimer b {
    color: #78350f !important;
}


/* ---------------------------------------------------------
   ERROR
--------------------------------------------------------- */

.error-box {
    background: #fef2f2;

    color: #991b1b;

    border: 1px solid #fecaca;

    border-left: 5px solid #dc2626;

    border-radius: 10px;

    padding: 18px;

    font-weight: 600;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 700px) {

    h1 {
        font-size: 27px !important;
    }

    h2 {
        font-size: 21px !important;
    }

    .header-box {
        padding: 20px !important;
    }

    .prediction-value {
        font-size: 25px;
    }

    .bmi-number {
        font-size: 26px;
    }

}

"""


# ============================================================
# 14. GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="AI/ML-Based Obesity Level Prediction and Personalized Habit Coaching System",
    css=css
) as app:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="header-box">

            <h1>
                AI/ML-Based Obesity Level Prediction
                and Personalized Habit Coaching System
            </h1>

            <p>
                A machine-learning based application for obesity
                level prediction and personalized lifestyle guidance.
            </p>

        </div>
        """
    )


    # ========================================================
    # ABOUT
    # ========================================================

    gr.Markdown("## About Application")

    gr.HTML(
        """
        <div class="info-box">

            <p>
            This application uses Machine Learning to predict
            obesity levels from demographic, eating and lifestyle
            information.
            </p>

            <p>
            A Random Forest classification model is trained using
            the UCI Obesity Dataset. The application also calculates
            BMI separately and provides simple personalized
            habit recommendations.
            </p>

        </div>
        """
    )


    # ========================================================
    # HOW TO ENTER VALUES
    # ========================================================

    gr.Markdown("## How to Enter Values")

    gr.HTML(
        """
        <div class="info-box">

            <ul>

                <li>
                    Enter the information that best describes you.
                </li>

                <li>
                    Enter height in metres.
                </li>

                <li>
                    Enter weight in kilograms.
                </li>

                <li>
                    Use the sliders for lifestyle-related values.
                </li>

                <li>
                    Click <b>Predict Obesity Level</b> to view your results.
                </li>

            </ul>

        </div>
        """
    )


    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

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


    # ========================================================
    # FAMILY & EATING HABITS
    # ========================================================

    gr.Markdown("## Family & Eating Habits")

    with gr.Row():

        family_history = gr.Dropdown(
            choices=["Yes", "No"],
            label="Family History of Overweight",
            value="No"
        )

        favc = gr.Dropdown(
            choices=["Yes", "No"],
            label="Frequent High-Calorie Food",
            value="No"
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
                "No",
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


    # ========================================================
    # PHYSICAL ACTIVITY & OTHER HABITS
    # ========================================================

    gr.Markdown("## Physical Activity & Other Habits")

    with gr.Row():

        smoke = gr.Dropdown(
            choices=["Yes", "No"],
            label="Smoking",
            value="No"
        )

        scc = gr.Dropdown(
            choices=["Yes", "No"],
            label="Calories Monitoring",
            value="No"
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
                "No",
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


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    gr.Markdown("")

    predict_button = gr.Button(
        "Predict Obesity Level",
        variant="primary"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    gr.Markdown("## Prediction Results")

    result_output = gr.HTML()


    # ========================================================
    # BMI
    # ========================================================

    gr.Markdown("### BMI Indicator")

    bmi_output = gr.HTML()


    # ========================================================
    # PROBABILITY
    # ========================================================

    gr.Markdown("### Prediction Probability")

    probability_output = gr.HTML()


    # ========================================================
    # UNDERSTANDING RESULTS
    # ========================================================

    gr.Markdown("## Understanding Results")

    gr.HTML(
        """
        <div class="info-box">

            <p>
            The predicted obesity level is generated by the
            Random Forest machine-learning model using the
            information entered above.
            </p>

            <p>
            BMI is calculated separately using height and weight:
            </p>

            <p style="font-size:17px;">

                <b>BMI = Weight (kg) / Height² (m²)</b>

            </p>

            <p>
            The probability values show how strongly the trained
            model associates the input with each obesity category.
            </p>

        </div>
        """
    )


    # ========================================================
    # HABIT COACH
    # ========================================================

    gr.Markdown("## Simple Habit Coach")

    habit_output = gr.Markdown(
        "Your habit suggestions will appear here."
    )


    # ========================================================
    # ACTION PLAN
    # ========================================================

    gr.Markdown("## Personalized Action Plan")

    action_output = gr.Markdown(
        "Your action plan will appear here."
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    gr.Markdown("## About AI/ML Model")

    gr.HTML(
        """
        <div class="info-box">

            <ul>

                <li>
                    <b>Dataset:</b> UCI Obesity Dataset
                </li>

                <li>
                    <b>Dataset ID:</b> 544
                </li>

                <li>
                    <b>Model:</b> Random Forest Classifier
                </li>

                <li>
                    <b>Number of Trees:</b> 300
                </li>

                <li>
                    <b>Train-Test Split:</b> 80:20
                </li>

                <li>
                    <b>Classification:</b> 7 obesity categories
                </li>

                <li>
                    <b>Framework:</b> Scikit-learn
                </li>

                <li>
                    <b>Interface:</b> Gradio
                </li>

            </ul>

        </div>
        """
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

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


    # ========================================================
    # BUTTON FUNCTION
    # ========================================================

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
