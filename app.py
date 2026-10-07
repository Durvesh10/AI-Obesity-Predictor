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

# Height and Weight are handled separately for BMI.
# NObeyesdad is the target variable.

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
# 6. COMPLETE MACHINE LEARNING PIPELINE
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


    # --------------------------------------------------------
    # High-Calorie Food
    # --------------------------------------------------------

    if favc == "Yes":

        recommendations.append(
            "Prefer healthy snacks and avoid frequent unhealthy eating between meals."
        )


    # --------------------------------------------------------
    # Vegetable Consumption
    # --------------------------------------------------------

    try:

        if float(fcvc) < 2:

            recommendations.append(
                "Increase vegetable consumption and include vegetables in your regular meals."
            )

    except:

        pass


    # --------------------------------------------------------
    # Number of Meals
    # --------------------------------------------------------

    try:

        if float(ncp) < 3:

            recommendations.append(
                "Maintain regular and balanced main meals instead of skipping meals."
            )

    except:

        pass


    # --------------------------------------------------------
    # Food Between Meals
    # --------------------------------------------------------

    if caec in [
        "Sometimes",
        "Frequently",
        "Always"
    ]:

        recommendations.append(
            "Prefer healthy snacks and avoid frequent unhealthy eating between meals."
        )


    # --------------------------------------------------------
    # Water Consumption
    # --------------------------------------------------------

    try:

        if float(ch2o) < 2:

            recommendations.append(
                "Increase your daily water intake and stay adequately hydrated."
            )

    except:

        pass


    # --------------------------------------------------------
    # Smoking
    # --------------------------------------------------------

    if smoke == "Yes":

        recommendations.append(
            "Consider reducing or avoiding smoking to support overall health."
        )


    # --------------------------------------------------------
    # Calorie Monitoring
    # --------------------------------------------------------

    if scc == "No":

        recommendations.append(
            "Consider monitoring calorie intake to better understand your eating habits."
        )


    # --------------------------------------------------------
    # Alcohol
    # --------------------------------------------------------

    if calc in [
        "Sometimes",
        "Frequently"
    ]:

        recommendations.append(
            "Limit alcohol consumption and maintain healthier beverage choices."
        )


    # --------------------------------------------------------
    # Physical Activity
    # --------------------------------------------------------

    try:

        if float(faf) < 2:

            recommendations.append(
                "Gradually increase physical activity such as walking, cycling or exercise."
            )

    except:

        pass


    # --------------------------------------------------------
    # Technology Usage
    # --------------------------------------------------------

    try:

        if float(tue) > 2:

            recommendations.append(
                "Reduce prolonged screen/device usage and take regular activity breaks."
            )

    except:

        pass


    # --------------------------------------------------------
    # Default Recommendation
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append(
            "Your current lifestyle inputs look relatively balanced. Continue maintaining healthy habits."
        )


    # --------------------------------------------------------
    # FORMAT OUTPUT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Food
    # --------------------------------------------------------

    if favc == "Yes":

        actions.append(
            "Reduce high-calorie and highly processed foods."
        )

    else:

        actions.append(
            "Continue maintaining balanced food choices."
        )


    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    try:

        if float(ch2o) < 2:

            actions.append(
                "Increase daily water consumption."
            )

        else:

            actions.append(
                "Continue maintaining adequate hydration."
            )

    except:

        actions.append(
            "Maintain adequate daily water intake."
        )


    # --------------------------------------------------------
    # Physical Activity
    # --------------------------------------------------------

    try:

        if float(faf) < 2:

            actions.append(
                "Add regular physical activity to your daily routine."
            )

        else:

            actions.append(
                "Continue your regular physical activity."
            )

    except:

        actions.append(
            "Maintain a consistent physical activity routine."
        )


    # --------------------------------------------------------
    # Screen Time
    # --------------------------------------------------------

    try:

        if float(tue) > 2:

            actions.append(
                "Reduce prolonged screen time and take movement breaks."
            )

        else:

            actions.append(
                "Continue maintaining reasonable screen/device usage."
            )

    except:

        actions.append(
            "Take regular breaks from prolonged screen usage."
        )


    # --------------------------------------------------------
    # Smoking
    # --------------------------------------------------------

    if smoke == "Yes":

        actions.append(
            "Work toward reducing or avoiding smoking."
        )

    else:

        actions.append(
            "Continue avoiding smoking."
        )


    # --------------------------------------------------------
    # Alcohol
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FORMAT ACTION PLAN
    # --------------------------------------------------------

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
        # Prepare User Input
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
        # Machine Learning Prediction
        # ----------------------------------------------------

        prediction = pipeline.predict(
            user_data
        )[0]


        probabilities = pipeline.predict_proba(
            user_data
        )[0]


        classes = pipeline.classes_


        # ----------------------------------------------------
        # Convert Prediction to User-Friendly Name
        # ----------------------------------------------------

        predicted_label = label_map.get(
            prediction,
            prediction
        )


        # ----------------------------------------------------
        # Risk Calculation
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
        # Prediction Output
        # ----------------------------------------------------

        prediction_output = f"""
## 🧠 Predicted Obesity Level

### **{predicted_label}**

The machine-learning model predicts your most likely obesity category based on the information provided.
"""


        # ----------------------------------------------------
        # BMI Category
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
        # BMI Output
        # ----------------------------------------------------

        bmi_output = f"""
### BMI: **{bmi}**

**BMI Category:** {bmi_category}
"""


        # ----------------------------------------------------
        # Risk Output
        # ----------------------------------------------------

        risk_output = f"""
### Estimated Risk

**{risk_percentage:.1f}%**

This value represents the model's estimated probability across the non-healthy obesity categories used in this application.
"""


        # ----------------------------------------------------
        # Create Charts
        # ----------------------------------------------------

        bmi_chart = create_bmi_gauge(
            bmi
        )


        probability_chart = create_probability_chart(
            probabilities,
            classes
        )


        # ----------------------------------------------------
        # Habit Coach
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
        # Action Plan
        # ----------------------------------------------------

        action_plan = generate_action_plan(
            favc,
            ch2o,
            smoke,
            calc,
            faf,
            tue
        )


        # ----------------------------------------------------
        # Return Everything
        # ----------------------------------------------------

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
# 16. CSS
# ============================================================

CSS = """

body {
    background: #f5f7fb;
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
    # MAIN TITLE
    # ========================================================

    gr.HTML("""

    <div style="
        text-align:center;
        width:100%;
        padding:20px 0 30px 0;
    ">

        <div style="
            font-size:36px;
            font-weight:800;
            line-height:1.25;
        ">

            🧠 AI/ML-Based Obesity Level Prediction
            and Personalized Habit Coaching System

        </div>

        <div style="
            font-size:19px;
            color:#64748b;
            margin-top:12px;
        ">

            Predict obesity level and receive personalized
            lifestyle suggestions

        </div>

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

            value="Male"

        )


        age = gr.Number(

            label="Age",

            value=22

        )


        height = gr.Number(

            label="Height (m)",

            value=1.70

        )


        weight = gr.Number(

            label="Weight (kg)",

            value=70

        )


    # ========================================================
    # EATING & LIFESTYLE
    # ========================================================

    gr.Markdown(
        "## 🍎 Eating & Lifestyle Habits"
    )


    with gr.Row():

        family_history = gr.Radio(

            choices=[
                "yes",
                "no"
            ],

            label="Family History with Overweight",

            value="yes"

        )


        favc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Frequent High-Calorie Food",

            value="No"

        )


        fcvc = gr.Slider(

            minimum=1,
            maximum=3,
            value=2,
            step=0.1,

            label="Vegetable Consumption Frequency"

        )


        ncp = gr.Slider(

            minimum=1,
            maximum=4,
            value=3,
            step=0.1,

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

            label="Food Between Meals",

            value="Sometimes"

        )


        smoke = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Smoking",

            value="No"

        )


        ch2o = gr.Slider(

            minimum=1,
            maximum=3,
            value=2,
            step=0.1,

            label="Daily Water Consumption"

        )


        scc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Monitor Calorie Consumption",

            value="No"

        )


    with gr.Row():

        faf = gr.Slider(

            minimum=0,
            maximum=3,
            value=1,
            step=0.1,

            label="Physical Activity Frequency"

        )


        tue = gr.Slider(

            minimum=0,
            maximum=2,
            value=1,
            step=0.1,

            label="Technology Usage Time"

        )


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
                "Public_Transportation",
                "Automobile",
                "Walking",
                "Motorbike",
                "Bike"
            ],

            label="Transportation",

            value="Public_Transportation"

        )


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
