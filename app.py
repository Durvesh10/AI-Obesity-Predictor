# ============================================
# AI OBESITY LEVEL PREDICTOR & HABIT COACH
# ============================================

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
from sklearn.metrics import accuracy_score


# ============================================
# LOAD DATASET
# ============================================

print("Loading UCI Obesity dataset...")

ds = fetch_ucirepo(id=544)

df = pd.concat(
    [ds.data.features, ds.data.targets],
    axis=1
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================
# PREPARE DATA
# ============================================

X = df.drop(
    columns=["NObeyesdad", "Height", "Weight"]
)

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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================
# RANDOM FOREST MODEL
# ============================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    )
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


print("Training model...")

model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)


# ============================================
# SETTINGS
# ============================================

HEALTHY = [
    "Insufficient_Weight",
    "Normal_Weight"
]


PRETTY = {
    "Insufficient_Weight": "Underweight",
    "Normal_Weight": "Normal Weight",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III"
}


# ============================================
# BMI FUNCTIONS
# ============================================

def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight"

    elif bmi < 25:
        return "Normal Weight"

    elif bmi < 30:
        return "Overweight"

    elif bmi < 35:
        return "Obesity Class I"

    elif bmi < 40:
        return "Obesity Class II"

    else:
        return "Obesity Class III"


def risk_level(risk):

    if risk < 0.30:
        return "Low"

    elif risk < 0.60:
        return "Moderate"

    else:
        return "High"


# ============================================
# INPUT CONVERSION FUNCTIONS
# ============================================

def convert_fcvc(value):

    mapping = {
        "Rarely": 1,
        "Sometimes": 2,
        "Usually": 3
    }

    return mapping[value]


def convert_ncp(value):

    mapping = {
        "1 main meal": 1,
        "2 main meals": 2,
        "3 main meals": 3,
        "4 or more main meals": 4
    }

    return mapping[value]


def convert_ch2o(value):

    mapping = {
        "Less than 1 L/day": 1,
        "1–2 L/day": 2,
        "More than 2 L/day": 3
    }

    return mapping[value]


def convert_faf(value):

    mapping = {
        "No regular activity": 0,
        "1–2 days/week": 1,
        "3–4 days/week": 2,
        "5+ days/week": 3
    }

    return mapping[value]


def convert_tue(value):

    mapping = {
        "Less than 1 hour/day": 0,
        "1–2 hours/day": 1,
        "More than 2 hours/day": 2
    }

    return mapping[value]


def convert_yes_no(value):

    mapping = {
        "Yes": "yes",
        "No": "no"
    }

    return mapping[value]


def convert_frequency(value):

    mapping = {
        "Never": "no",
        "Sometimes": "Sometimes",
        "Frequently": "Frequently",
        "Always": "Always"
    }

    return mapping[value]


def convert_transport(value):

    mapping = {
        "Walking": "Walking",
        "Bike": "Bike",
        "Public Transportation": "Public_Transportation",
        "Motorbike": "Motorbike",
        "Automobile": "Automobile"
    }

    return mapping[value]


# ============================================
# CREATE MODEL INPUT ROW
# ============================================

def make_row(
    gender,
    age,
    family,
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

    row = pd.DataFrame([{

        "Gender": gender,

        "Age": age,

        "family_history_with_overweight":
            convert_yes_no(family),

        "FAVC":
            convert_yes_no(favc),

        "FCVC":
            convert_fcvc(fcvc),

        "NCP":
            convert_ncp(ncp),

        "CAEC":
            convert_frequency(caec),

        "SMOKE":
            convert_yes_no(smoke),

        "CH2O":
            convert_ch2o(ch2o),

        "SCC":
            convert_yes_no(scc),

        "FAF":
            convert_faf(faf),

        "TUE":
            convert_tue(tue),

        "CALC":
            convert_frequency(calc),

        "MTRANS":
            convert_transport(mtrans)

    }])

    return row[X.columns]


# ============================================
# RISK CALCULATION
# ============================================

def risk_of(row):

    probabilities = model.predict_proba(row)[0]

    risk = sum(

        probability

        for class_name, probability
        in zip(
            model.classes_,
            probabilities
        )

        if class_name not in HEALTHY
    )

    return risk


# ============================================
# HABIT COACH
# ============================================

def habit_coach(row):

    recommendations = []

    faf = float(
        row["FAF"].iloc[0]
    )

    fcvc = float(
        row["FCVC"].iloc[0]
    )

    favc = str(
        row["FAVC"].iloc[0]
    ).lower()

    caec = str(
        row["CAEC"].iloc[0]
    ).lower()

    ch2o = float(
        row["CH2O"].iloc[0]
    )

    tue = float(
        row["TUE"].iloc[0]
    )


    if faf < 2:

        recommendations.append(
            "🏃 **Increase physical activity:** "
            "Try walking, cycling, or exercising "
            "for 20–30 minutes regularly."
        )

    else:

        recommendations.append(
            "🏃 **Good activity level:** "
            "Continue staying physically active."
        )


    if fcvc < 2:

        recommendations.append(
            "🥗 **Eat more vegetables:** "
            "Add vegetables or salad to your main meals."
        )

    else:

        recommendations.append(
            "🥗 **Good vegetable intake:** "
            "Continue including vegetables in your meals."
        )


    if favc == "yes":

        recommendations.append(
            "🍔 **Reduce high-calorie foods:** "
            "Limit junk food, fried food, and "
            "highly processed foods."
        )

    else:

        recommendations.append(
            "🍔 **Good choice:** "
            "Continue limiting high-calorie foods."
        )


    if ch2o < 2:

        recommendations.append(
            "💧 **Drink more water:** "
            "Try to maintain regular hydration "
            "throughout the day."
        )

    else:

        recommendations.append(
            "💧 **Good hydration:** "
            "Continue maintaining regular water intake."
        )


    if caec != "no":

        recommendations.append(
            "🍎 **Control snacking:** "
            "Reduce unnecessary snacks between meals."
        )

    else:

        recommendations.append(
            "🍎 **Good snacking habit:** "
            "Continue keeping unnecessary snacking low."
        )


    if tue > 1:

        recommendations.append(
            "📱 **Reduce screen time:** "
            "Take regular breaks and include more "
            "physical movement."
        )

    else:

        recommendations.append(
            "📱 **Good screen-time control:** "
            "Continue taking regular movement breaks."
        )


    return recommendations


# ============================================
# ACTION PLAN
# ============================================

def create_action_plan(row, bmi, prediction):

    actions = []

    faf = float(
        row["FAF"].iloc[0]
    )

    fcvc = float(
        row["FCVC"].iloc[0]
    )

    favc = str(
        row["FAVC"].iloc[0]
    ).lower()

    ch2o = float(
        row["CH2O"].iloc[0]
    )


    if faf < 2:

        actions.append(
            "🏃 Start with 20–30 minutes of "
            "walking or light exercise regularly."
        )

    else:

        actions.append(
            "🏃 Maintain your current physical activity."
        )


    if fcvc < 2:

        actions.append(
            "🥗 Add vegetables or salad to your main meals."
        )

    else:

        actions.append(
            "🥗 Continue including vegetables in your meals."
        )


    if favc == "yes":

        actions.append(
            "🍔 Reduce frequent high-calorie "
            "and highly processed foods."
        )

    else:

        actions.append(
            "🍔 Continue limiting high-calorie foods."
        )


    if ch2o < 2:

        actions.append(
            "💧 Increase water intake gradually "
            "and stay hydrated."
        )

    else:

        actions.append(
            "💧 Continue maintaining regular water intake."
        )


    return actions


# ============================================
# PROBABILITY CHART
# ============================================

def create_probability_chart(
    probabilities,
    classes
):

    probability_data = []


    for class_name, probability in zip(
        classes,
        probabilities
    ):

        probability_data.append({

            "class":
                PRETTY.get(
                    class_name,
                    class_name
                ),

            "probability":
                probability * 100

        })


    probability_data = sorted(

        probability_data,

        key=lambda x:
            x["probability"],

        reverse=True
    )


    labels = [
        item["class"]
        for item in probability_data
    ]

    values = [
        item["probability"]
        for item in probability_data
    ]


    fig = go.Figure()


    fig.add_trace(

        go.Bar(

            x=values,

            y=labels,

            orientation="h",

            text=[
                f"{value:.1f}%"
                for value in values
            ],

            textposition="outside",

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Probability: %{x:.1f}%"
                "<extra></extra>"
            )

        )
    )


    fig.update_layout(

        title={
            "text": "AI Prediction Probability",
            "x": 0.5,
            "xanchor": "center"
        },

        xaxis={
            "title": "Probability (%)",

            "range": [
                0,
                max(
                    100,
                    max(values) + 12
                )
            ]
        },

        yaxis={
            "title": ""
        },

        height=450,

        template="plotly_white",

        margin={
            "l": 30,
            "r": 80,
            "t": 80,
            "b": 50
        },

        showlegend=False
    )


    return fig


# ============================================
# BMI GAUGE
# ============================================

def create_bmi_chart(bmi):

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=bmi,

            title={
                "text":
                    "Body Mass Index (BMI)"
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
                        "range": [30, 35],
                        "name": "Obesity I"
                    },

                    {
                        "range": [35, 40],
                        "name": "Obesity II"
                    },

                    {
                        "range": [40, 50],
                        "name": "Obesity III"
                    }

                ]
            }
        )
    )


    fig.update_layout(

        height=350,

        template="plotly_white"
    )


    return fig


# ============================================
# MAIN PREDICTION
# ============================================

def run_prediction(

    gender,
    age,
    family,
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
    mtrans,
    height,
    weight

):

    bmi = weight / (height ** 2)

    bmi_cat = bmi_category(bmi)


    row = make_row(

        gender,
        age,
        family,
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
    )


    prediction = model.predict(row)[0]

    probabilities = model.predict_proba(row)[0]

    risk = risk_of(row)

    risk_cat = risk_level(risk)


    probability_chart = create_probability_chart(

        probabilities,

        model.classes_
    )


    bmi_chart = create_bmi_chart(bmi)


    recommendations = habit_coach(row)


    actions = create_action_plan(

        row,
        bmi,
        prediction
    )


    return {

        "prediction":
            PRETTY.get(
                prediction,
                prediction
            ),

        "bmi":
            round(
                bmi,
                2
            ),

        "bmi_category":
            bmi_cat,

        "risk":
            round(
                risk * 100,
                2
            ),

        "risk_category":
            risk_cat,

        "probability_chart":
            probability_chart,

        "bmi_chart":
            bmi_chart,

        "recommendations":
            recommendations,

        "actions":
            actions
    }


# ============================================
# DISPLAY RESULTS
# ============================================

def predict_and_display(

    gender,
    age,
    family,
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
    mtrans,
    height,
    weight

):

    try:

        if height is None or height <= 0:

            raise ValueError(
                "Please enter a valid height."
            )


        if weight is None or weight <= 0:

            raise ValueError(
                "Please enter a valid weight."
            )


        if age is None or age <= 0:

            raise ValueError(
                "Please enter a valid age."
            )


        result = run_prediction(

            gender,
            age,
            family,
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
            mtrans,
            height,
            weight
        )


        # ----------------------------
        # PREDICTION
        # ----------------------------

        prediction_text = f"""
## 🎯 Predicted Obesity Level

# {result["prediction"]}

The AI model's most likely classification based on the information provided.
"""


        # ----------------------------
        # BMI
        # ----------------------------

        bmi_text = f"""
## ⚖️ BMI

# {result["bmi"]}

**Category:** {result["bmi_category"]}
"""


        # ----------------------------
        # RISK
        # ----------------------------

        risk_text = f"""
## 📈 Overall Risk

# {result["risk"]:.1f}%

**Risk Level:** {result["risk_category"]}

This represents the model's estimated probability of being in an overweight or obesity category.
"""


        # ----------------------------
        # HABIT COACH
        # IMPORTANT:
        # No duplicate heading here.
        # ----------------------------

        recommendations_text = """
Based on your lifestyle answers:

"""


        for recommendation in result["recommendations"]:

            recommendations_text += (
                f"- {recommendation}\n\n"
            )


        # ----------------------------
        # ACTION PLAN
        # IMPORTANT:
        # No duplicate heading here.
        # ----------------------------

        action_text = ""


        for i, action in enumerate(
            result["actions"],
            1
        ):

            action_text += (
                f"**{i}.** {action}\n\n"
            )


        return (

            prediction_text,

            bmi_text,

            risk_text,

            result["bmi_chart"],

            result["probability_chart"],

            recommendations_text,

            action_text
        )


    except Exception as e:

        error_message = f"""
## ❌ Prediction Error

Something went wrong while generating the prediction.

**Error details:**

`{str(e)}`

Please check your inputs and try again.
"""


        return (

            error_message,

            error_message,

            error_message,

            None,

            None,

            error_message,

            error_message
        )


# ============================================
# CSS
# ============================================

CSS = """

body {

    background: #f5f7fb;

}


.gradio-container {

    max-width: 1200px !important;

    margin: auto !important;

}


.main-title {

    text-align: center;

    font-size: 34px;

    font-weight: 700;

    margin-bottom: 5px;

}


.subtitle {

    text-align: center;

    color: #64748b;

    font-size: 16px;

    margin-bottom: 25px;

}

"""


# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(

    title="AI Obesity Predictor & Habit Coach",

    css=CSS,

    theme=gr.themes.Soft()

) as demo:


    gr.Markdown("""

        <div class="main-title">

            🧠 AI Obesity Level Predictor

        </div>

        <div class="subtitle">

            Predict obesity level and receive
            personalized lifestyle suggestions

        </div>

    """)


    # ========================================
    # PERSONAL INFORMATION
    # ========================================

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

            value=21,

            minimum=10,

            maximum=100
        )


        family = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="Family history of overweight",

            value="No"
        )


    # ========================================
    # BODY MEASUREMENTS
    # ========================================

    gr.Markdown(
        "## 📏 Body Measurements"
    )


    with gr.Row():

        height = gr.Number(

            label="Height (metres)",

            value=1.70,

            minimum=1.0,

            maximum=2.5
        )


        weight = gr.Number(

            label="Weight (kg)",

            value=65,

            minimum=20,

            maximum=250
        )


    # ========================================
    # EATING HABITS
    # ========================================

    gr.Markdown(
        "## 🍽️ Eating Habits"
    )


    gr.Markdown(
        "**Answer according to your usual daily habits.**"
    )


    with gr.Row():

        favc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="🍔 Do you frequently eat high-calorie food?",

            value="No"
        )


        fcvc = gr.Radio(

            choices=[
                "Rarely",
                "Sometimes",
                "Usually"
            ],

            label="🥗 How often do you eat vegetables?",

            value="Sometimes"
        )


    with gr.Row():

        ncp = gr.Dropdown(

            choices=[
                "1 main meal",
                "2 main meals",
                "3 main meals",
                "4 or more main meals"
            ],

            label="🍽️ How many main meals do you usually eat?",

            value="3 main meals"
        )


        caec = gr.Radio(

            choices=[
                "Never",
                "Sometimes",
                "Frequently",
                "Always"
            ],

            label="🍎 How often do you snack between meals?",

            value="Sometimes"
        )


    # ========================================
    # LIFESTYLE
    # ========================================

    gr.Markdown(
        "## 🏃 Lifestyle"
    )


    with gr.Row():

        smoke = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="🚬 Do you smoke?",

            value="No"
        )


        ch2o = gr.Radio(

            choices=[
                "Less than 1 L/day",
                "1–2 L/day",
                "More than 2 L/day"
            ],

            label="💧 How much water do you drink daily?",

            value="1–2 L/day"
        )


        scc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            label="📊 Do you monitor your calorie intake?",

            value="No"
        )


    with gr.Row():

        faf = gr.Radio(

            choices=[
                "No regular activity",
                "1–2 days/week",
                "3–4 days/week",
                "5+ days/week"
            ],

            label="🏋️ How often are you physically active?",

            value="1–2 days/week"
        )


        tue = gr.Radio(

            choices=[
                "Less than 1 hour/day",
                "1–2 hours/day",
                "More than 2 hours/day"
            ],

            label="📱 How much time do you spend using technology/screen?",

            value="1–2 hours/day"
        )


    with gr.Row():

        calc = gr.Radio(

            choices=[
                "Never",
                "Sometimes",
                "Frequently",
                "Always"
            ],

            label="🍺 How often do you consume alcohol?",

            value="Never"
        )


        mtrans = gr.Dropdown(

            choices=[
                "Walking",
                "Bike",
                "Public Transportation",
                "Motorbike",
                "Automobile"
            ],

            label="🚶 What is your main mode of transportation?",

            value="Public Transportation"
        )


    # ========================================
    # PREDICT BUTTON
    # ========================================

    predict_button = gr.Button(

        "🔍 Predict Obesity Level",

        variant="primary",

        size="lg"
    )


    gr.Markdown("---")


    # ========================================
    # PREDICTION RESULTS
    # ========================================

    gr.Markdown(
        "## 📊 Prediction Results"
    )


    with gr.Row():

        result_prediction = gr.Markdown("""

            ### 🎯 Predicted Level

            **Waiting for prediction...**

        """)


        result_bmi = gr.Markdown("""

            ### ⚖️ BMI

            **Waiting for prediction...**

        """)


        result_risk = gr.Markdown("""

            ### 📈 Overall Risk

            **Waiting for prediction...**

        """)


    with gr.Row():

        bmi_output = gr.Plot(
            label="BMI Analysis"
        )


        probability_output = gr.Plot(
            label="Prediction Probabilities"
        )


    # ========================================
    # HABIT COACH
    # ========================================

    gr.Markdown(
        "## 🧠 Simple Habit Coach"
    )


    habit_output = gr.Markdown(

        "Your personalized suggestions "
        "will appear here."

    )


    # ========================================
    # ACTION PLAN
    # ========================================

    gr.Markdown(
        "## 🏃 Your Action Plan"
    )


    action_output = gr.Markdown(

        "Your personalized action plan "
        "will appear here."

    )


    # ========================================
    # BUTTON FUNCTION
    # ========================================

    predict_button.click(

        fn=predict_and_display,

        inputs=[

            gender,
            age,
            family,
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
            mtrans,
            height,
            weight

        ],

        outputs=[

            result_prediction,

            result_bmi,

            result_risk,

            bmi_output,

            probability_output,

            habit_output,

            action_output

        ]

    )


# ============================================
# APPLICATION START
# ============================================

print(
    "Application created successfully!"
)


demo.launch(

    server_name="0.0.0.0",

    server_port=int(
        os.environ.get(
            "PORT",
            7860
        )
    )

)
