
import os
import pandas as pd
import numpy as np
import gradio as gr
import plotly.graph_objects as go

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD UCI DATASET
# ============================================================

obesity = fetch_ucirepo(id=544)

X_raw = obesity.data.features
y = obesity.data.targets.iloc[:, 0]

df = X_raw.copy()
df["NObeyesdad"] = y


# ============================================================
# 2. PREPARE DATA
# ============================================================

# Height and Weight are used separately for BMI calculation.
# They are NOT directly used as ML input features.

X = df.drop(
    columns=["NObeyesdad", "Height", "Weight"]
)

y = df["NObeyesdad"]


# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================

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
# 4. DATA PREPROCESSING
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
# 5. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)


# ============================================================
# 6. MACHINE LEARNING PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 9. USER-FRIENDLY CLASS NAMES
# ============================================================

label_map = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III"
}


# ============================================================
# 10. BMI CALCULATION
# ============================================================

def calculate_bmi(height, weight):

    try:

        height = float(height)
        weight = float(weight)

        if height <= 0 or weight <= 0:
            return None

        bmi = weight / (height ** 2)

        return round(bmi, 2)

    except:

        return None


# ============================================================
# 11. BMI GAUGE
# ============================================================

def create_bmi_gauge(bmi):

    if bmi is None:
        return go.Figure()

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=bmi,

            title={
                "text": "BMI"
            },

            gauge={
                "axis": {
                    "range": [10, 50]
                },

                "bar": {
                    "thickness": 0.25
                },

                "steps": [
                    {
                        "range": [10, 18.5],
                        "name": "Underweight"
                    },

                    {
                        "range": [18.5, 25],
                        "name": "Normal"
                    },

                    {
                        "range": [25, 30],
                        "name": "Overweight"
                    },

                    {
                        "range": [30, 50],
                        "name": "Obesity"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=300,

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    return fig


# ============================================================
# 12. PROBABILITY CHART
# ============================================================

def create_probability_chart(
    probabilities,
    classes
):

    display_classes = [
        label_map.get(c, c)
        for c in classes
    ]

    probability_values = [
        float(p) * 100
        for p in probabilities
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=display_classes,

                y=probability_values,

                text=[
                    f"{p:.1f}%"
                    for p in probability_values
                ],

                textposition="auto"
            )
        ]
    )

    fig.update_layout(
        title="Obesity Class Probability",

        xaxis_title="Obesity Category",

        yaxis_title="Probability (%)",

        yaxis={
            "range": [0, 100]
        },

        height=400,

        margin=dict(
            l=40,
            r=30,
            t=60,
            b=100
        )
    )

    return fig


# ============================================================
# 13. HABIT COACH
# ============================================================

def generate_habit_coach(
    favc,
    fcvc,
    ncp,
    caec,
    ch2o,
    smoke,
    scc,
    calc,
    faf,
    tue
):

    recommendations = []


    # High-calorie food

    if favc == "Yes":

        recommendations.append(
            "Prefer healthier food choices and reduce frequent consumption of high-calorie foods."
        )


    # Vegetable consumption

    try:

        if float(fcvc) < 2:

            recommendations.append(
                "Increase vegetable consumption and include vegetables regularly in your meals."
            )

    except:

        pass


    # Number of meals

    try:

        if float(ncp) < 3:

            recommendations.append(
                "Maintain regular and balanced main meals instead of frequently skipping meals."
            )

    except:

        pass


    # Food between meals

    if caec in [
        "Sometimes",
        "Frequently",
        "Always"
    ]:

        recommendations.append(
            "Prefer healthy snacks and avoid frequent unhealthy eating between meals."
        )


    # Water consumption

    try:

        if float(ch2o) < 2:

            recommendations.append(
                "Increase your daily water intake and stay adequately hydrated."
            )

    except:

        pass


    # Smoking

    if smoke == "Yes":

        recommendations.append(
            "Consider reducing or avoiding smoking to support overall health."
        )


    # Calorie monitoring

    if scc == "No":

        recommendations.append(
            "Consider monitoring calorie intake to better understand your eating habits."
        )


    # Alcohol

    if calc in [
        "Sometimes",
        "Frequently"
    ]:

        recommendations.append(
            "Limit alcohol consumption and choose healthier beverage options."
        )


    # Physical activity

    try:

        if float(faf) < 2:

            recommendations.append(
                "Gradually increase physical activity such as walking, cycling or exercise."
            )

    except:

        pass


    # Technology usage

    try:

        if float(tue) > 2:

            recommendations.append(
                "Reduce prolonged screen/device usage and take regular movement breaks."
            )

    except:

        pass


    # Default recommendation

    if not recommendations:

        recommendations.append(
            "Your current lifestyle inputs appear relatively balanced. Continue maintaining healthy habits."
        )


    # Separate bullet points

    recommendations_text = """
### Personalized Suggestions

Based on your lifestyle answers:

"""

    for recommendation in recommendations:

        recommendations_text += (
            f"• {recommendation}\n\n"
        )


    return recommendations_text


# ============================================================
# 14. ACTION PLAN
# ============================================================

def generate_action_plan(
    favc,
    ch2o,
    smoke,
    calc,
    faf,
    tue
):

    actions = []


    # Food

    if favc == "Yes":

        actions.append(
            "Reduce frequent consumption of high-calorie and highly processed foods."
        )

    else:

        actions.append(
            "Continue maintaining balanced and nutritious food choices."
        )


    # Water

    try:

        if float(ch2o) < 2:

            actions.append(
                "Increase daily water consumption gradually."
            )

        else:

            actions.append(
                "Continue maintaining adequate hydration."
            )

    except:

        actions.append(
            "Maintain adequate daily water intake."
        )


    # Physical activity

    try:

        if float(faf) < 2:

            actions.append(
                "Add regular physical activity such as walking, cycling or exercise."
            )

        else:

            actions.append(
                "Continue your existing physical activity routine."
            )

    except:

        actions.append(
            "Maintain a consistent physical activity routine."
        )


    # Screen time

    try:

        if float(tue) > 2:

            actions.append(
                "Reduce prolonged screen time and take regular movement breaks."
            )

        else:

            actions.append(
                "Continue maintaining reasonable screen/device usage."
            )

    except:

        actions.append(
            "Take regular breaks from prolonged screen usage."
        )


    # Smoking

    if smoke == "Yes":

        actions.append(
            "Work toward reducing or avoiding smoking."
        )

    else:

        actions.append(
            "Continue avoiding smoking."
        )


    # Alcohol

    if calc in [
        "Sometimes",
        "Frequently"
    ]:

        actions.append(
            "Limit alcohol consumption."
        )

    else:

        actions.append(
            "Continue maintaining responsible beverage choices."
        )


    # Format action plan

    action_text = """
### Your Personalized Action Plan

"""

    for number, action in enumerate(
        actions,
        start=1
    ):

        action_text += (
            f"{number}. {action}\n\n"
        )


    return action_text


# ============================================================
# 15. MAIN PREDICTION FUNCTION
# ============================================================

def predict_obesity(
    gender,
    age,
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

    try:

        # ----------------------------------------------------
        # BMI
        # ----------------------------------------------------

        bmi = calculate_bmi(
            height,
            weight
        )


        if bmi is None:

            return (
                "Please enter valid height and weight values.",
                "",
                "",
                go.Figure(),
                go.Figure(),
                "Please enter valid values.",
                "Please enter valid values."
            )


        # ----------------------------------------------------
        # USER INPUT
        # ----------------------------------------------------

        user_data = pd.DataFrame(
            [{
                "Gender": gender,

                "Age": float(age),

                "family_history_with_overweight":
                    family_history,

                "FAVC": favc,

                "FCVC": float(fcvc),

                "NCP": float(ncp),

                "CAEC": caec,

                "SMOKE": smoke,

                "CH2O": float(ch2o),

                "SCC": scc,

                "FAF": float(faf),

                "TUE": float(tue),

                "CALC": calc,

                "MTRANS": mtrans
            }]
        )


        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        prediction = pipeline.predict(
            user_data
        )[0]


        probabilities = pipeline.predict_proba(
            user_data
        )[0]


        classes = pipeline.classes_


        # ----------------------------------------------------
        # FRIENDLY LABEL
        # ----------------------------------------------------

        predicted_label = label_map.get(
            prediction,
            prediction
        )


        # ----------------------------------------------------
        # RISK CALCULATION
        # ----------------------------------------------------

        healthy_classes = [
            "Insufficient_Weight",
            "Normal_Weight"
        ]


        risk_probability = sum(

            probabilities[i]

            for i, c in enumerate(classes)

            if c not in healthy_classes

        )


        risk_percentage = (
            risk_probability * 100
        )


        # ----------------------------------------------------
        # PREDICTION OUTPUT
        # ----------------------------------------------------

        prediction_output = f"""
## 🧠 Predicted Obesity Level

### **{predicted_label}**

The machine-learning model predicts this category based on the physical and lifestyle information provided.
"""


        # ----------------------------------------------------
        # BMI CATEGORY
        # ----------------------------------------------------

        if bmi < 18.5:

            bmi_category = "Underweight"

        elif bmi < 25:

            bmi_category = "Normal"

        elif bmi < 30:

            bmi_category = "Overweight"

        else:

            bmi_category = "Obesity"


        # ----------------------------------------------------
        # BMI OUTPUT
        # ----------------------------------------------------

        bmi_output = f"""
### BMI: **{bmi}**

**BMI Category:** {bmi_category}

BMI is calculated separately using:

**BMI = Weight (kg) / Height² (m²)**
"""


        # ----------------------------------------------------
        # RISK OUTPUT
        # ----------------------------------------------------

        risk_output = f"""
### Estimated Risk Indicator

**{risk_percentage:.1f}%**

This represents the model's estimated probability across the non-healthy obesity categories used in this application.

It is **not a medically validated risk score or diagnosis**.
"""


        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        bmi_chart = create_bmi_gauge(
            bmi
        )


        probability_chart = create_probability_chart(
            probabilities,
            classes
        )


        # ----------------------------------------------------
        # HABIT COACH
        # ----------------------------------------------------

        habit_coach = generate_habit_coach(
            favc,
            fcvc,
            ncp,
            caec,
            ch2o,
            smoke,
            scc,
            calc,
            faf,
            tue
        )


        # ----------------------------------------------------
        # ACTION PLAN
        # ----------------------------------------------------

        action_plan = generate_action_plan(
            favc,
            ch2o,
            smoke,
            calc,
            faf,
            tue
        )


        return (
            prediction_output,
            bmi_output,
            risk_output,
            bmi_chart,
            probability_chart,
            habit_coach,
            action_plan
        )


    except Exception as e:

        return (
            f"Error: {str(e)}",
            "",
            "",
            go.Figure(),
            go.Figure(),
            "Unable to generate suggestions.",
            "Unable to generate action plan."
        )


# ============================================================
# 16. SIMPLE CSS
# ============================================================

CSS = """

body {
    background-color: #f5f7fb;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding-top: 20px !important;
}

"""


# ============================================================
# 17. GRADIO APPLICATION
# ============================================================

with gr.Blocks(

    title="AI/ML-Based Obesity Level Prediction and Personalized Habit Coaching System",

    css=CSS,

    theme=gr.themes.Soft()

) as demo:


    # ========================================================
    # TITLE
    # ========================================================

    gr.HTML("""

    <div style="
        text-align:center;
        width:100%;
        padding:20px 0 25px 0;
        color:#0f172a;
    ">

        <div style="
            font-size:36px;
            font-weight:800;
            line-height:1.25;
            color:#0f172a;
        ">

            🧠 AI/ML-Based Obesity Level Prediction
            and Personalized Habit Coaching System

        </div>

        <div style="
            font-size:19px;
            color:#475569;
            margin-top:12px;
        ">

            Predict obesity level and receive personalized
            lifestyle suggestions

        </div>

    </div>

    """)


    # ========================================================
    # ABOUT APPLICATION
    # ========================================================

    gr.Markdown(
        "## ℹ️ About This Application"
    )


    # INLINE STYLING ONLY
    # This prevents Chrome from changing the text color.

    gr.HTML("""

    <div style="
        background:#ffffff;
        color:#1e293b;
        border:1px solid #dbe3ec;
        border-radius:12px;
        padding:20px 24px;
        margin:10px 0 20px 0;
        box-shadow:0 2px 8px rgba(15,23,42,0.05);
    ">

        <h3 style="
            color:#0f172a;
            margin-top:0;
        ">
            What does this application do?
        </h3>

        <p style="color:#1e293b;">
            This application uses a machine-learning model to
            estimate an obesity category from physical and
            lifestyle-related information.
        </p>

        <p style="color:#1e293b;">
            It also calculates BMI separately and provides
            probability visualizations, simple habit suggestions
            and a personalized action plan.
        </p>

        <p style="color:#1e293b;">
            <b>Machine Learning Model:</b>
            Random Forest Classifier with 300 decision trees.
        </p>

        <p style="color:#1e293b;">
            <b>Dataset:</b>
            UCI Obesity Dataset.
        </p>

        <p style="color:#1e293b;">
            <b>Learning Type:</b>
            Supervised Machine Learning – Multiclass Classification.
        </p>

        <p style="color:#1e293b;">
            <b>Important:</b>
            The results are intended for educational and
            informational purposes and should not be treated
            as a medical diagnosis.
        </p>

    </div>

    """)


    # ========================================================
    # HOW TO ENTER VALUES
    # ========================================================

    gr.Markdown(
        "## 📖 How to Enter the Values"
    )


    gr.HTML("""

    <div style="
        background:#f8fafc;
        color:#1e293b;
        border:1px solid #dbe3ec;
        border-radius:12px;
        padding:20px 24px;
        margin:8px 0 20px 0;
    ">

        <h3 style="
            color:#0f172a;
            margin-top:0;
        ">
            🔢 Understanding the Numerical Scales
        </h3>

        <p style="color:#1e293b;">
            Some questions use numerical scales instead of direct
            measurements. These values represent the coding used
            by the dataset.
        </p>


        <hr style="
            border:none;
            border-top:1px solid #cbd5e1;
            margin:18px 0;
        ">


        <h4 style="color:#0f172a;">
            FCVC – Vegetable Consumption Frequency
        </h4>

        <ul style="color:#1e293b;">

            <li style="color:#1e293b;">
                <b>1</b> → Rarely consume vegetables
            </li>

            <li style="color:#1e293b;">
                <b>2</b> → Sometimes / moderate consumption
            </li>

            <li style="color:#1e293b;">
                <b>3</b> → Frequently consume vegetables
            </li>

        </ul>

        <p style="color:#1e293b;">
            Example: <b>2.3</b> represents a value between
            2 and 3. Use the slider to select an approximate value.
        </p>


        <hr style="
            border:none;
            border-top:1px solid #cbd5e1;
            margin:18px 0;
        ">


        <h4 style="color:#0f172a;">
            NCP – Number of Main Meals
        </h4>

        <ul style="color:#1e293b;">

            <li style="color:#1e293b;">
                <b>1</b> → About one main meal per day
            </li>

            <li style="color:#1e293b;">
                <b>2</b> → About two main meals per day
            </li>

            <li style="color:#1e293b;">
                <b>3</b> → About three main meals per day
            </li>

            <li style="color:#1e293b;">
                <b>4</b> → Four or more main meals per day
            </li>

        </ul>

        <p style="color:#1e293b;">
            Example: If you normally eat three main meals,
            choose approximately <b>3</b>.
        </p>


        <hr style="
            border:none;
            border-top:1px solid #cbd5e1;
            margin:18px 0;
        ">


        <h4 style="color:#0f172a;">
            CH2O – Daily Water Consumption
        </h4>

        <ul style="color:#1e293b;">

            <li style="color:#1e293b;">
                <b>1</b> → Low water consumption
            </li>

            <li style="color:#1e293b;">
                <b>2</b> → Moderate water consumption
            </li>

            <li style="color:#1e293b;">
                <b>3</b> → High water consumption
            </li>

        </ul>

        <p style="color:#1e293b;">
            <b>Important:</b> These are dataset scale values,
            <b>not litres</b>. Do not enter your water intake
            directly in litres.
        </p>


        <hr style="
            border:none;
            border-top:1px solid #cbd5e1;
            margin:18px 0;
        ">


        <h4 style="color:#0f172a;">
            FAF – Physical Activity Frequency
        </h4>

        <ul style="color:#1e293b;">

            <li style="color:#1e293b;">
                <b>0</b> → Little or no physical activity
            </li>

            <li style="color:#1e293b;">
                <b>1</b> → Low physical activity
            </li>

            <li style="color:#1e293b;">
                <b>2</b> → Moderate physical activity
            </li>

            <li style="color:#1e293b;">
                <b>3</b> → High physical activity
            </li>

        </ul>

        <p style="color:#1e293b;">
            Example: If you exercise regularly but not very
            frequently, a value around <b>1–2</b> may be appropriate.
        </p>


        <hr style="
            border:none;
            border-top:1px solid #cbd5e1;
            margin:18px 0;
        ">


        <h4 style="color:#0f172a;">
            TUE – Technology Usage Time
        </h4>

        <ul style="color:#1e293b;">

            <li style="color:#1e293b;">
                <b>0</b> → Low device/screen usage
            </li>

            <li style="color:#1e293b;">
                <b>1</b> → Moderate device/screen usage
            </li>

            <li style="color:#1e293b;">
                <b>2</b> → High device/screen usage
            </li>

        </ul>

        <p style="color:#1e293b;">
            This is a dataset scale and should not be interpreted
            as an exact number of hours.
        </p>


        <hr style="
            border:none;
            border-top:1px solid #cbd5e1;
            margin:18px 0;
        ">


        <h4 style="color:#0f172a;">
            Physical Measurements
        </h4>

        <p style="color:#1e293b;">
            <b>Age:</b> Enter your age in years.
        </p>

        <p style="color:#1e293b;">
            <b>Height:</b> Enter height in metres.
            Example: 170 cm = <b>1.70 m</b>.
        </p>

        <p style="color:#1e293b;">
            <b>Weight:</b> Enter body weight in kilograms.
            Example: 70 kg = <b>70</b>.
        </p>

    </div>

    """)


    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

    gr.Markdown(
        "## 👤 Personal Information"
    )


    with gr.Row():

        gender = gr.Dropdown(

            choices=[
                "Male",
                "Female"
            ],

            label="Gender",

            info="Select your gender.",

            value="Male"

        )


        age = gr.Number(

            label="Age (years)",

            info="Enter your age in completed years.",

            value=22

        )


        height = gr.Number(

            label="Height (metres)",

            info="Example: 170 cm = 1.70 m.",

            value=1.70

        )


        weight = gr.Number(

            label="Weight (kg)",

            info="Enter your current body weight in kilograms.",

            value=70

        )


    # ========================================================
    # FAMILY & EATING HABITS
    # ========================================================

    gr.Markdown(
        "## 🧬 Family & Eating Habits"
    )


    with gr.Row():

        family_history = gr.Radio(

            choices=[
                "yes",
                "no"
            ],

            label="Family History with Overweight",

            info="Has anyone in your family had overweight/obesity?",

            value="yes"

        )


        favc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Frequent High-Calorie Food",

            info="Do you frequently eat high-calorie foods?",

            value="No"

        )


        fcvc = gr.Slider(

            minimum=1,
            maximum=3,
            value=2,
            step=0.1,

            label="Vegetable Consumption Frequency",

            info="1 = low, 2 = moderate, 3 = frequent."

        )


        ncp = gr.Slider(

            minimum=1,
            maximum=4,
            value=3,
            step=0.1,

            label="Number of Main Meals",

            info="Approximately how many main meals do you have per day?"

        )


    with gr.Row():

        caec = gr.Dropdown(

            choices=[
                "No",
                "Sometimes",
                "Frequently",
                "Always"
            ],

            label="Food Between Meals",

            info="How often do you eat between your main meals?",

            value="Sometimes"

        )


        smoke = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Smoking",

            info="Do you currently smoke?",

            value="No"

        )


        ch2o = gr.Slider(

            minimum=1,
            maximum=3,
            value=2,
            step=0.1,

            label="Daily Water Consumption",

            info="1 = low, 2 = moderate, 3 = high. Dataset scale, not litres."

        )


        scc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Monitor Calorie Consumption",

            info="Do you monitor your calorie intake?",

            value="No"

        )


    # ========================================================
    # ACTIVITY & OTHER HABITS
    # ========================================================

    gr.Markdown(
        "## 🏃 Physical Activity & Other Habits"
    )


    with gr.Row():

        faf = gr.Slider(

            minimum=0,
            maximum=3,
            value=1,
            step=0.1,

            label="Physical Activity Frequency",

            info="0 = little/none, 1 = low, 2 = moderate, 3 = high."

        )


        tue = gr.Slider(

            minimum=0,
            maximum=2,
            value=1,
            step=0.1,

            label="Technology Usage Time",

            info="0 = low, 1 = moderate, 2 = high. Dataset scale."

        )


        calc = gr.Dropdown(

            choices=[
                "No",
                "Sometimes",
                "Frequently",
                "Always"
            ],

            label="Alcohol Consumption",

            info="How frequently do you consume alcohol?",

            value="Sometimes"

        )


        mtrans = gr.Dropdown(

            choices=[
                "Public_Transportation",
                "Automobile",
                "Walking",
                "Motorbike",
                "Bike"
            ],

            label="Transportation",

            info="Select your usual mode of transportation.",

            value="Public_Transportation"

        )


    # ========================================================
    # QUICK INPUT REMINDER
    # ========================================================

    gr.HTML("""

    <div style="
        background:#ffffff;
        color:#1e293b;
        border:1px solid #dbe3ec;
        border-radius:12px;
        padding:20px 24px;
        margin:10px 0 20px 0;
    ">

        <h3 style="
            color:#0f172a;
            margin-top:0;
        ">
            ✅ Quick Input Reminder
        </h3>

        <ul>

            <li style="color:#1e293b;">
                Height → enter in <b>metres</b>, e.g. 1.70
            </li>

            <li style="color:#1e293b;">
                Weight → enter in <b>kilograms</b>, e.g. 70
            </li>

            <li style="color:#1e293b;">
                FCVC → <b>1–3 scale</b> for vegetable consumption
            </li>

            <li style="color:#1e293b;">
                NCP → approximately <b>1–4 main meals</b>
            </li>

            <li style="color:#1e293b;">
                CH2O → <b>1–3 dataset scale</b>, not litres
            </li>

            <li style="color:#1e293b;">
                FAF → <b>0–3 scale</b> for physical activity
            </li>

            <li style="color:#1e293b;">
                TUE → <b>0–2 scale</b> for technology usage
            </li>

        </ul>

    </div>

    """)


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    predict_button = gr.Button(

        "🔍 Predict Obesity Level",

        variant="primary",

        size="lg"

    )


    # ========================================================
    # RESULTS
    # ========================================================

    gr.Markdown(
        "## 📊 Prediction Results"
    )


    with gr.Row():

        prediction_output = gr.Markdown(
            "Your prediction will appear here."
        )


        bmi_output = gr.Markdown(
            "Your BMI will appear here."
        )


        risk_output = gr.Markdown(
            "Your estimated risk will appear here."
        )


    with gr.Row():

        bmi_chart = gr.Plot(
            label="BMI Indicator"
        )


        probability_chart = gr.Plot(
            label="Prediction Probability"
        )


    # ========================================================
    # RESULT EXPLANATION
    # ========================================================

    gr.HTML("""

    <div style="
        background:#ffffff;
        color:#1e293b;
        border:1px solid #dbe3ec;
        border-radius:12px;
        padding:20px 24px;
        margin:10px 0 20px 0;
    ">

        <h3 style="
            color:#0f172a;
            margin-top:0;
        ">
            📌 Understanding Your Results
        </h3>

        <p style="color:#1e293b;">
            <b>Predicted Obesity Level:</b>
            This is the category predicted by the Random Forest
            machine-learning model.
        </p>

        <p style="color:#1e293b;">
            <b>BMI:</b>
            BMI is calculated separately using your height and weight.
            It is not directly used as an input feature in the current
            machine-learning model.
        </p>

        <p style="color:#1e293b;">
            <b>Probability Chart:</b>
            The chart shows the model's estimated probability for
            each of the seven possible categories.
        </p>

        <p style="color:#1e293b;">
            <b>Estimated Risk Indicator:</b>
            This combines the model probabilities of the
            non-healthy categories. It is an application-specific
            model indicator and is not a clinically validated risk score.
        </p>

    </div>

    """)


    # ========================================================
    # HABIT COACH
    # ========================================================

    gr.Markdown(
        "## 🧠 Simple Habit Coach"
    )


    habit_output = gr.Markdown(
        "Your personalized suggestions will appear here."
    )


    # ========================================================
    # ACTION PLAN
    # ========================================================

    gr.Markdown(
        "## 🏃 Your Action Plan"
    )


    action_output = gr.Markdown(
        "Your personalized action plan will appear here."
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    gr.Markdown(
        "## 🤖 About the AI/ML Model"
    )


    gr.HTML("""

    <div style="
        background:#ffffff;
        color:#1e293b;
        border:1px solid #dbe3ec;
        border-radius:12px;
        padding:20px 24px;
        margin:10px 0 20px 0;
    ">

        <h3 style="
            color:#0f172a;
            margin-top:0;
        ">
            Machine Learning Details
        </h3>

        <p style="color:#1e293b;">
            This application uses a <b>Random Forest Classifier</b>
            trained on the UCI Obesity Dataset.
        </p>

        <ul>

            <li style="color:#1e293b;">
                <b>Learning Type:</b> Supervised Learning
            </li>

            <li style="color:#1e293b;">
                <b>Task:</b> Multiclass Classification
            </li>

            <li style="color:#1e293b;">
                <b>Algorithm:</b> Random Forest Classifier
            </li>

            <li style="color:#1e293b;">
                <b>Number of Trees:</b> 300
            </li>

            <li style="color:#1e293b;">
                <b>Train/Test Split:</b> 80% / 20%
            </li>

            <li style="color:#1e293b;">
                <b>Categorical Data:</b> One-Hot Encoded
            </li>

            <li style="color:#1e293b;">
                <b>Output:</b> Seven obesity-related categories
            </li>

        </ul>

        <p style="color:#1e293b;">
            The <b>Habit Coach</b> and <b>Action Plan</b> are
            rule-based components. They are not generated by the
            Random Forest model.
        </p>

    </div>

    """)


    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.HTML("""

    <div style="
        background:#fff7ed;
        color:#431407;
        border:1px solid #fdba74;
        border-radius:12px;
        padding:22px 24px;
        margin-top:30px;
        margin-bottom:20px;
    ">

        <h2 style="
            color:#7c2d12;
            margin-top:0;
        ">
            ⚠️ Disclaimer
        </h2>

        <p style="color:#431407;">
            This application is developed for
            <b>educational, demonstration and informational purposes only</b>.
        </p>

        <p style="color:#431407;">
            The predictions generated by this application are based
            on a machine-learning model trained on the UCI Obesity
            Dataset and should <b>not</b> be considered a medical
            diagnosis, clinical assessment or professional medical advice.
        </p>

        <p style="color:#431407;">
            BMI and the model's estimated probabilities have limitations
            and may not accurately represent an individual's complete
            health condition.
        </p>

        <p style="color:#431407;">
            The Habit Coach and Action Plan provide general lifestyle
            suggestions and are not a substitute for advice from a
            qualified doctor, dietitian, nutritionist or other healthcare
            professional.
        </p>

        <p style="color:#431407;">
            If you have concerns about your weight, nutrition, physical
            activity or overall health, consult a qualified healthcare
            professional.
        </p>

        <p style="color:#431407;">
            <b>
            By using this application, you acknowledge that its results
            are informational and should be interpreted with appropriate
            professional guidance.
            </b>
        </p>

    </div>

    """)


    # ========================================================
    # CONNECT BUTTON TO PREDICTION FUNCTION
    # ========================================================

    predict_button.click(

        fn=predict_obesity,

        inputs=[

            gender,
            age,
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

            prediction_output,
            bmi_output,
            risk_output,
            bmi_chart,
            probability_chart,
            habit_output,
            action_output

        ]

    )


# ============================================================
# 18. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )

    )
