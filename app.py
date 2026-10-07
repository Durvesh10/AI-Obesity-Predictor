
import os
import pandas as pd
import gradio as gr

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
# 2. PREPARE DATA
# ============================================================

# Height and Weight are used separately for BMI.
# They are not used as ML input features.

X = df.drop(columns=["NObeyesdad", "Height", "Weight"])
y = df["NObeyesdad"]


# ============================================================
# 3. FEATURES
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
# 4. PREPROCESSING
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
# 5. MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)


# ============================================================
# 6. PIPELINE
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

pipeline.fit(X_train, y_train)


# ============================================================
# 9. LABELS
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
# 10. BMI
# ============================================================

def calculate_bmi(height, weight):

    try:
        height = float(height)
        weight = float(weight)

        if height <= 0 or weight <= 0:
            return None

        return round(weight / (height ** 2), 2)

    except:
        return None


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
# 11. SIMPLE BMI INDICATOR
# ============================================================

def create_bmi_html(bmi):

    if bmi is None:
        return "<p>Please enter valid height and weight.</p>"

    category = get_bmi_category(bmi)

    position = ((bmi - 10) / 40) * 100
    position = max(0, min(100, position))

    return f"""
    <div style="
        padding:15px;
        text-align:center;
        font-family:Arial;
    ">

        <h3>BMI Indicator</h3>

        <div style="
            font-size:32px;
            font-weight:bold;
            margin:10px;
        ">
            {bmi}
        </div>

        <div style="
            font-size:18px;
            font-weight:bold;
            margin-bottom:15px;
        ">
            {category}
        </div>

        <div style="
            position:relative;
            width:100%;
            height:35px;
            display:flex;
            border-radius:6px;
            overflow:hidden;
            border:1px solid #999;
        ">

            <div style="
                width:21.25%;
                background:#b9d7f5;
                padding-top:8px;
                font-size:11px;
            ">
                Underweight
            </div>

            <div style="
                width:16.25%;
                background:#b9e6c3;
                padding-top:8px;
                font-size:11px;
            ">
                Normal
            </div>

            <div style="
                width:12.5%;
                background:#f5df9a;
                padding-top:8px;
                font-size:11px;
            ">
                Overweight
            </div>

            <div style="
                width:50%;
                background:#f2b5b5;
                padding-top:8px;
                font-size:11px;
            ">
                Obesity
            </div>

            <div style="
                position:absolute;
                left:{position}%;
                top:-2px;
                transform:translateX(-50%);
                font-size:18px;
                font-weight:bold;
            ">
                ▼
            </div>

        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            font-size:11px;
            margin-top:5px;
        ">
            <span>10</span>
            <span>18.5</span>
            <span>25</span>
            <span>30</span>
            <span>50</span>
        </div>

        <p style="font-size:12px;">
            BMI = Weight (kg) / Height² (m²)
        </p>

    </div>
    """


# ============================================================
# 12. SIMPLE PROBABILITY CHART
# ============================================================

def create_probability_html(probabilities, classes):

    rows = ""

    for probability, class_name in zip(
        probabilities,
        classes
    ):

        label = label_map.get(
            class_name,
            class_name
        )

        percentage = float(probability) * 100

        rows += f"""
        <div style="
            margin:12px 0;
            font-family:Arial;
        ">

            <div style="
                display:flex;
                justify-content:space-between;
                font-size:13px;
                margin-bottom:4px;
            ">

                <span>{label}</span>

                <span>{percentage:.1f}%</span>

            </div>

            <div style="
                width:100%;
                height:18px;
                background:#e5e5e5;
                border-radius:5px;
                overflow:hidden;
            ">

                <div style="
                    width:{percentage:.2f}%;
                    height:100%;
                    background:#3b82f6;
                ">
                </div>

            </div>

        </div>
        """

    return f"""
    <div style="
        padding:15px;
        font-family:Arial;
    ">

        <h3 style="text-align:center;">
            Prediction Probability
        </h3>

        <p style="
            text-align:center;
            font-size:13px;
        ">
            Estimated probability for each category
        </p>

        {rows}

    </div>
    """


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

    if favc == "Yes":
        recommendations.append(
            "Reduce frequent consumption of high-calorie foods."
        )

    try:
        if float(fcvc) < 2:
            recommendations.append(
                "Increase vegetable consumption."
            )
    except:
        pass

    try:
        if float(ncp) < 3:
            recommendations.append(
                "Try to maintain regular and balanced main meals."
            )
    except:
        pass

    if caec in ["Sometimes", "Frequently", "Always"]:
        recommendations.append(
            "Choose healthier snacks between meals."
        )

    try:
        if float(ch2o) < 2:
            recommendations.append(
                "Increase water consumption and maintain hydration."
            )
    except:
        pass

    if smoke == "Yes":
        recommendations.append(
            "Consider reducing or avoiding smoking."
        )

    if scc == "No":
        recommendations.append(
            "Consider monitoring your calorie intake."
        )

    if calc in ["Sometimes", "Frequently", "Always"]:
        recommendations.append(
            "Limit alcohol consumption."
        )

    try:
        if float(faf) < 2:
            recommendations.append(
                "Gradually increase physical activity."
            )
    except:
        pass

    try:
        if float(tue) > 1.5:
            recommendations.append(
                "Reduce prolonged screen/device usage and take breaks."
            )
    except:
        pass

    if not recommendations:
        recommendations.append(
            "Your current lifestyle inputs appear relatively balanced. "
            "Continue maintaining healthy habits."
        )

    output = "### 🧠 Personalized Suggestions\n\n"

    for item in recommendations:
        output += f"- {item}\n\n"

    return output


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

    if favc == "Yes":
        actions.append(
            "Reduce frequent consumption of high-calorie and processed foods."
        )
    else:
        actions.append(
            "Continue maintaining balanced food choices."
        )

    try:
        if float(ch2o) < 2:
            actions.append(
                "Gradually increase daily water consumption."
            )
        else:
            actions.append(
                "Continue maintaining adequate hydration."
            )
    except:
        actions.append(
            "Maintain adequate daily water intake."
        )

    try:
        if float(faf) < 2:
            actions.append(
                "Add regular physical activity such as walking or exercise."
            )
        else:
            actions.append(
                "Continue your physical activity routine."
            )
    except:
        actions.append(
            "Maintain a consistent physical activity routine."
        )

    try:
        if float(tue) > 1.5:
            actions.append(
                "Reduce prolonged screen time and take movement breaks."
            )
        else:
            actions.append(
                "Continue maintaining reasonable screen usage."
            )
    except:
        actions.append(
            "Take regular breaks from prolonged screen usage."
        )

    if smoke == "Yes":
        actions.append(
            "Work toward reducing or avoiding smoking."
        )
    else:
        actions.append(
            "Continue avoiding smoking."
        )

    if calc in ["Sometimes", "Frequently", "Always"]:
        actions.append(
            "Limit alcohol consumption."
        )
    else:
        actions.append(
            "Continue maintaining responsible beverage choices."
        )

    output = "### 🏃 Personalized Action Plan\n\n"

    for i, action in enumerate(actions, 1):
        output += f"{i}. {action}\n\n"

    return output


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

        # BMI
        bmi = calculate_bmi(height, weight)

        if bmi is None:
            return (
                "❌ Please enter valid height and weight.",
                "",
                "",
                "",
                "",
                "",
                ""
            )

        # User data
        user_data = pd.DataFrame([{

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

        }])

        # Prediction
        prediction = pipeline.predict(
            user_data
        )[0]

        probabilities = pipeline.predict_proba(
            user_data
        )[0]

        classes = pipeline.classes_

        predicted_label = label_map.get(
            prediction,
            prediction
        )

        # Risk
        healthy_classes = [
            "Insufficient_Weight",
            "Normal_Weight"
        ]

        risk_probability = sum(
            probabilities[i]
            for i, class_name in enumerate(classes)
            if class_name not in healthy_classes
        )

        risk_percentage = risk_probability * 100

        # Outputs
        prediction_output = f"""
### 🧠 Predicted Obesity Level

# {predicted_label}

The machine-learning model predicts this category from
the physical and lifestyle information provided.
"""

        bmi_category = get_bmi_category(bmi)

        bmi_output = f"""
### ⚖️ BMI

# {bmi}

**BMI Category:** {bmi_category}

BMI is calculated separately using height and weight.
"""

        risk_output = f"""
### 📊 Estimated Risk Indicator

# {risk_percentage:.1f}%

This is the model's estimated probability across the
non-healthy obesity categories.

**Note:** This is not a medically validated risk score.
"""

        bmi_html = create_bmi_html(bmi)

        probability_html = create_probability_html(
            probabilities,
            classes
        )

        habit_output = generate_habit_coach(
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

        action_output = generate_action_plan(
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
            bmi_html,
            probability_html,
            habit_output,
            action_output
        )

    except Exception as e:

        return (
            f"❌ Error: {str(e)}",
            "",
            "",
            "",
            "",
            "",
            ""
        )


# ============================================================
# 16. SIMPLE CSS
# ============================================================

CSS = """

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.app-title {
    text-align: center;
    padding: 10px 0 20px 0;
}

.app-title h1 {
    font-size: 30px;
    margin-bottom: 8px;
}

.app-title p {
    font-size: 16px;
}

.info-box {
    padding: 12px 16px;
    margin: 5px 0 15px 0;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: white;
}

.info-box h3 {
    margin-top: 0;
}

.small-text {
    font-size: 13px;
}

.disclaimer {
    padding: 15px;
    margin-top: 25px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
}

"""


# ============================================================
# 17. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="AI/ML-Based Obesity Level Prediction",
    css=CSS,
    theme=gr.themes.Soft()
) as demo:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    gr.HTML("""
    <div class="app-title">

        <h1>
            🧠 AI/ML-Based Obesity Level Prediction
        </h1>

        <p>
            Predict obesity level and receive personalized
            lifestyle suggestions.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    gr.Markdown("## ℹ️ About the Application")

    gr.HTML("""
    <div class="info-box">

        <h3>What does this application do?</h3>

        <p>
            This application uses a machine-learning model
            to predict an obesity category using physical
            and lifestyle information.
        </p>

        <p>
            It also calculates BMI separately and provides
            prediction probabilities, habit suggestions
            and an action plan.
        </p>

        <p>
            <b>Model:</b> Random Forest Classifier
            with 300 trees.
        </p>

        <p>
            <b>Dataset:</b> UCI Obesity Dataset.
        </p>

        <p>
            <b>Task:</b> Supervised multiclass classification.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # HOW TO ENTER VALUES
    # --------------------------------------------------------

    gr.Markdown("## 📖 How to Enter Values")

    gr.HTML("""
    <div class="info-box">

        <p>
            <b>FCVC:</b> Vegetable consumption frequency
            — 1 = low, 2 = moderate, 3 = high.
        </p>

        <p>
            <b>NCP:</b> Number of main meals
            — approximately 1 to 4.
        </p>

        <p>
            <b>CH2O:</b> Water consumption
            — dataset scale from low to high.
        </p>

        <p>
            <b>FAF:</b> Physical activity frequency
            — 0 = little/none, 1 = low, 2 = moderate, 3 = high.
        </p>

        <p>
            <b>TUE:</b> Technology usage
            — dataset scale from low to high.
        </p>

        <p>
            <b>Height:</b> Enter in metres.
            Example: 170 cm = 1.70 m.
        </p>

        <p>
            <b>Weight:</b> Enter in kilograms.
            Example: 70 kg.
        </p>

        <p class="small-text">
            <b>Note:</b> The numerical lifestyle values are
            dataset scales and are not always direct physical
            measurements.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    gr.Markdown("## 👤 Personal Information")

    with gr.Row():

        gender = gr.Dropdown(
            choices=["Male", "Female"],
            value="Male",
            label="Gender"
        )

        age = gr.Number(
            value=22,
            label="Age (years)"
        )

        height = gr.Number(
            value=1.70,
            label="Height (metres)",
            info="Example: 1.70"
        )

        weight = gr.Number(
            value=70,
            label="Weight (kg)"
        )


    # --------------------------------------------------------
    # FAMILY AND EATING HABITS
    # --------------------------------------------------------

    gr.Markdown("## 🧬 Family & Eating Habits")

    with gr.Row():

        family_history = gr.Radio(
            choices=["yes", "no"],
            value="yes",
            label="Family History with Overweight"
        )

        favc = gr.Radio(
            choices=["Yes", "No"],
            value="No",
            label="Frequent High-Calorie Food"
        )

        fcvc = gr.Slider(
            minimum=1,
            maximum=3,
            value=2,
            step=0.1,
            label="Vegetable Consumption Frequency",
            info="1 = low, 2 = moderate, 3 = high"
        )

        ncp = gr.Slider(
            minimum=1,
            maximum=4,
            value=3,
            step=0.1,
            label="Number of Main Meals",
            info="Approximately 1–4"
        )


    with gr.Row():

        caec = gr.Dropdown(
            choices=[
                "No",
                "Sometimes",
                "Frequently",
                "Always"
            ],
            value="Sometimes",
            label="Food Between Meals"
        )

        smoke = gr.Radio(
            choices=["Yes", "No"],
            value="No",
            label="Smoking"
        )

        ch2o = gr.Slider(
            minimum=1,
            maximum=3,
            value=2,
            step=0.1,
            label="Water Consumption",
            info="Dataset scale: low → high"
        )

        scc = gr.Radio(
            choices=["Yes", "No"],
            value="No",
            label="Monitor Calorie Consumption"
        )


    # --------------------------------------------------------
    # ACTIVITY AND OTHER HABITS
    # --------------------------------------------------------

    gr.Markdown("## 🏃 Physical Activity & Other Habits")

    with gr.Row():

        faf = gr.Slider(
            minimum=0,
            maximum=3,
            value=1,
            step=0.1,
            label="Physical Activity Frequency",
            info="0 = little/none, 3 = high"
        )

        tue = gr.Slider(
            minimum=0,
            maximum=2,
            value=1,
            step=0.1,
            label="Technology Usage Time",
            info="Dataset scale: low → high"
        )

        calc = gr.Dropdown(
            choices=[
                "No",
                "Sometimes",
                "Frequently",
                "Always"
            ],
            value="Sometimes",
            label="Alcohol Consumption"
        )

        mtrans = gr.Dropdown(
            choices=[
                "Public_Transportation",
                "Automobile",
                "Walking",
                "Motorbike",
                "Bike"
            ],
            value="Public_Transportation",
            label="Transportation"
        )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predict_button = gr.Button(
        "🔍 Predict Obesity Level",
        variant="primary",
        size="lg"
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    gr.Markdown("## 📊 Prediction Results")

    with gr.Row():

        prediction_output = gr.Markdown(
            "Prediction will appear here."
        )

        bmi_output = gr.Markdown(
            "BMI will appear here."
        )

        risk_output = gr.Markdown(
            "Risk indicator will appear here."
        )


    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    with gr.Row():

        bmi_result = gr.HTML("""
        <div style="text-align:center;padding:20px;">
            <h3>BMI Indicator</h3>
            <p>Your BMI indicator will appear here.</p>
        </div>
        """)

        probability_result = gr.HTML("""
        <div style="text-align:center;padding:20px;">
            <h3>Prediction Probability</h3>
            <p>Prediction probabilities will appear here.</p>
        </div>
        """)


    # --------------------------------------------------------
    # UNDERSTANDING RESULTS
    # --------------------------------------------------------

    gr.Markdown("## 📌 Understanding Your Results")

    gr.HTML("""
    <div class="info-box">

        <p>
            <b>Predicted Obesity Level:</b>
            Category predicted by the Random Forest model.
        </p>

        <p>
            <b>BMI:</b>
            Calculated separately from height and weight.
            BMI is not used as an ML input in this implementation.
        </p>

        <p>
            <b>Prediction Probability:</b>
            Estimated probability of each of the seven categories.
        </p>

        <p>
            <b>Estimated Risk:</b>
            Sum of model probabilities for the non-healthy categories.
        </p>

        <p class="small-text">
            This is not a clinically validated medical risk score.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # HABIT COACH
    # --------------------------------------------------------

    gr.Markdown("## 🧠 Simple Habit Coach")

    habit_output = gr.Markdown(
        "Your personalized suggestions will appear here."
    )


    # --------------------------------------------------------
    # ACTION PLAN
    # --------------------------------------------------------

    gr.Markdown("## 🏃 Personalized Action Plan")

    action_output = gr.Markdown(
        "Your action plan will appear here."
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    gr.Markdown("## 🤖 About the AI/ML Model")

    gr.HTML("""
    <div class="info-box">

        <ul>

            <li><b>Learning:</b> Supervised Learning</li>

            <li><b>Task:</b> Multiclass Classification</li>

            <li><b>Algorithm:</b> Random Forest Classifier</li>

            <li><b>Trees:</b> 300</li>

            <li><b>Train/Test Split:</b> 80% / 20%</li>

            <li><b>Encoding:</b> One-Hot Encoding</li>

            <li><b>Output:</b> Seven obesity categories</li>

        </ul>

        <p class="small-text">
            The Habit Coach and Action Plan are rule-based
            components and are not generated by the Random Forest model.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    gr.Markdown("## ⚠️ Disclaimer")

    gr.HTML("""
    <div class="disclaimer">

        <p>
            This application is developed for
            <b>educational, demonstration and informational purposes only.</b>
        </p>

        <p>
            The predictions are generated by a machine-learning
            model trained on the UCI Obesity Dataset and should
            not be considered a medical diagnosis or professional
            medical advice.
        </p>

        <p>
            BMI and model probabilities have limitations and may
            not represent an individual's complete health condition.
        </p>

        <p>
            The Habit Coach and Action Plan provide general
            lifestyle suggestions and are not a substitute for
            advice from a qualified healthcare professional.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # CONNECT BUTTON
    # --------------------------------------------------------

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
            bmi_result,
            probability_result,
            habit_output,
            action_output
        ]
    )


# ============================================================
# 18. LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        )
    )
