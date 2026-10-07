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

# Height and Weight are used separately for BMI.
# They are NOT used as ML input features.

X = df.drop(
    columns=["NObeyesdad", "Height", "Weight"]
)

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
# 5. RANDOM FOREST MODEL
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

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 9. LABEL MAP
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
# 11. BMI CATEGORY
# ============================================================

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
# 12. BMI HTML INDICATOR
# ============================================================

def create_bmi_html(bmi):

    if bmi is None:

        return """
        <div class="chart-card">

            <h3>BMI Indicator</h3>

            <p>
                Please enter valid height and weight values.
            </p>

        </div>
        """


    category = get_bmi_category(bmi)


    # Position marker on 10–50 scale
    position = ((bmi - 10) / 40) * 100

    position = max(
        0,
        min(100, position)
    )


    return f"""

    <div class="chart-card">

        <h3>BMI Indicator</h3>

        <div class="bmi-number">
            {bmi}
        </div>

        <div class="bmi-category">
            {category}
        </div>


        <div class="bmi-scale">

            <div class="bmi-part underweight">
                Underweight
            </div>

            <div class="bmi-part normal">
                Normal
            </div>

            <div class="bmi-part overweight">
                Overweight
            </div>

            <div class="bmi-part obesity">
                Obesity
            </div>

        </div>


        <div class="bmi-marker-container">

            <div
                class="bmi-marker"
                style="left: {position}%;">
                ▼
            </div>

        </div>


        <div class="bmi-values">

            <span>10</span>
            <span>18.5</span>
            <span>25</span>
            <span>30</span>
            <span>50</span>

        </div>


        <p class="chart-note">

            BMI = Weight (kg) / Height² (m²)

        </p>

    </div>

    """


# ============================================================
# 13. PROBABILITY HTML CHART
# ============================================================

def create_probability_html(
    probabilities,
    classes
):

    rows = ""


    for probability, class_name in zip(
        probabilities,
        classes
    ):

        label = label_map.get(
            class_name,
            class_name
        )


        percentage = float(
            probability
        ) * 100


        rows += f"""

        <div class="probability-row">

            <div class="probability-header">

                <span>
                    {label}
                </span>

                <span>
                    {percentage:.1f}%
                </span>

            </div>


            <div class="probability-track">

                <div
                    class="probability-bar"
                    style="width: {percentage:.2f}%;">
                </div>

            </div>

        </div>

        """


    return f"""

    <div class="chart-card probability-card">

        <h3>
            Prediction Probability
        </h3>

        <p class="chart-note">
            Estimated probability for each obesity category
        </p>

        {rows}

    </div>

    """


# ============================================================
# 14. HABIT COACH
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


    # High calorie food
    if favc == "Yes":

        recommendations.append(
            "Reduce frequent consumption of high-calorie foods "
            "and prefer balanced food choices."
        )


    # Vegetables
    try:

        if float(fcvc) < 2:

            recommendations.append(
                "Increase vegetable consumption and include "
                "vegetables regularly in meals."
            )

    except:
        pass


    # Meals
    try:

        if float(ncp) < 3:

            recommendations.append(
                "Try to maintain regular and balanced main meals "
                "instead of frequently skipping meals."
            )

    except:
        pass


    # Between meals
    if caec in [
        "Sometimes",
        "Frequently",
        "Always"
    ]:

        recommendations.append(
            "Choose healthier snacks and avoid frequent "
            "unhealthy eating between meals."
        )


    # Water
    try:

        if float(ch2o) < 2:

            recommendations.append(
                "Increase water consumption and maintain "
                "adequate hydration."
            )

    except:
        pass


    # Smoking
    if smoke == "Yes":

        recommendations.append(
            "Consider reducing or avoiding smoking "
            "to support overall health."
        )


    # Calorie monitoring
    if scc == "No":

        recommendations.append(
            "Consider monitoring calorie intake to better "
            "understand your eating habits."
        )


    # Alcohol
    if calc in [
        "Sometimes",
        "Frequently",
        "Always"
    ]:

        recommendations.append(
            "Limit alcohol consumption and prefer healthier "
            "beverage choices."
        )


    # Physical activity
    try:

        if float(faf) < 2:

            recommendations.append(
                "Gradually increase physical activity such as "
                "walking, cycling or exercise."
            )

    except:
        pass


    # Technology usage
    try:

        if float(tue) > 1.5:

            recommendations.append(
                "Reduce prolonged screen/device usage and "
                "take regular movement breaks."
            )

    except:
        pass


    # Default
    if not recommendations:

        recommendations.append(
            "Your current lifestyle inputs appear relatively "
            "balanced. Continue maintaining healthy habits."
        )


    output = """
## 🧠 Personalized Suggestions

"""


    for item in recommendations:

        output += (
            f"• {item}\n\n"
        )


    return output


# ============================================================
# 15. ACTION PLAN
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
            "Reduce frequent consumption of high-calorie "
            "and highly processed foods."
        )

    else:

        actions.append(
            "Continue maintaining balanced and nutritious "
            "food choices."
        )


    # Water
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


    # Activity
    try:

        if float(faf) < 2:

            actions.append(
                "Add regular physical activity such as walking, "
                "cycling or exercise."
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

        if float(tue) > 1.5:

            actions.append(
                "Reduce prolonged screen time and take "
                "regular movement breaks."
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
        "Frequently",
        "Always"
    ]:

        actions.append(
            "Limit alcohol consumption."
        )

    else:

        actions.append(
            "Continue maintaining responsible beverage choices."
        )


    output = """
## 🏃 Your Personalized Action Plan

"""


    for number, action in enumerate(
        actions,
        start=1
    ):

        output += (
            f"{number}. {action}\n\n"
        )


    return output


# ============================================================
# 16. MAIN PREDICTION FUNCTION
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
                "❌ Please enter valid height and weight values.",
                "",
                "",
                "",
                "",
                "Please enter valid values.",
                "Please enter valid values."
            )


        # ----------------------------------------------------
        # USER DATA
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
        # RISK
        # ----------------------------------------------------

        healthy_classes = [
            "Insufficient_Weight",
            "Normal_Weight"
        ]


        risk_probability = sum(

            probabilities[i]

            for i, class_name in enumerate(classes)

            if class_name not in healthy_classes

        )


        risk_percentage = (
            risk_probability * 100
        )


        # ----------------------------------------------------
        # PREDICTION OUTPUT
        # ----------------------------------------------------

        prediction_output = f"""

## 🧠 Predicted Obesity Level

# {predicted_label}

The machine-learning model predicts this category
from the physical and lifestyle information provided.

"""


        # ----------------------------------------------------
        # BMI OUTPUT
        # ----------------------------------------------------

        bmi_cat = get_bmi_category(
            bmi
        )


        bmi_output = f"""

## ⚖️ BMI

### {bmi}

**BMI Category:** {bmi_cat}

BMI is calculated separately using your height and weight.

"""


        # ----------------------------------------------------
        # RISK OUTPUT
        # ----------------------------------------------------

        risk_output = f"""

## 📊 Estimated Risk Indicator

### {risk_percentage:.1f}%

This represents the model's estimated probability
across the non-healthy obesity categories.

**Important:** This is not a medically validated
risk score or medical diagnosis.

"""


        # ----------------------------------------------------
        # CHROME-SAFE BMI
        # ----------------------------------------------------

        bmi_result = create_bmi_html(
            bmi
        )


        # ----------------------------------------------------
        # CHROME-SAFE PROBABILITY
        # ----------------------------------------------------

        probability_result = create_probability_html(
            probabilities,
            classes
        )


        # ----------------------------------------------------
        # HABIT COACH
        # ----------------------------------------------------

        habit_result = generate_habit_coach(
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

        action_result = generate_action_plan(
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
            bmi_result,
            probability_result,
            habit_result,
            action_result
        )


    except Exception as e:

        return (
            f"❌ Error: {str(e)}",
            "",
            "",
            "",
            "",
            "Unable to generate suggestions.",
            "Unable to generate action plan."
        )


# ============================================================
# 17. CHROME-FRIENDLY CSS
# ============================================================

CSS = """

/* ============================================================
   MAIN PAGE
   ============================================================ */

.gradio-container {

    max-width: 1200px !important;

    margin: auto !important;

    padding: 20px !important;

    font-family:
        Arial,
        Helvetica,
        sans-serif !important;
}


/* ============================================================
   TITLE
   ============================================================ */

.app-title {

    width: 100%;

    box-sizing: border-box;

    text-align: center;

    padding: 28px 20px;

    margin-bottom: 25px;

    background: #ffffff !important;

    border: 1px solid #d1d5db;

    border-radius: 12px;
}


.app-title h1 {

    margin: 0;

    color: #111827 !important;

    font-size: 34px;

    font-weight: 800;

    line-height: 1.3;
}


.app-title p {

    margin-top: 12px;

    color: #374151 !important;

    font-size: 18px;

    line-height: 1.5;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.app-info-card,
.quick-reminder,
.disclaimer-card {

    background: #ffffff !important;

    color: #111827 !important;

    border: 1px solid #d1d5db;

    border-radius: 12px;

    padding: 20px 24px;

    margin: 10px 0 20px 0;

    box-sizing: border-box;
}


.app-info-card h3,
.app-info-card h4,
.quick-reminder h3,
.disclaimer-card h2 {

    color: #111827 !important;

    margin-top: 0;
}


.app-info-card p,
.app-info-card li,
.quick-reminder li,
.disclaimer-card p {

    color: #374151 !important;

    line-height: 1.6;
}


.app-info-card hr {

    border: none;

    border-top: 1px solid #d1d5db;

    margin: 18px 0;
}


/* ============================================================
   CHART CARDS
   ============================================================ */

.chart-card {

    width: 100%;

    min-height: 300px;

    box-sizing: border-box;

    background: #ffffff !important;

    color: #111827 !important;

    border: 1px solid #d1d5db;

    border-radius: 12px;

    padding: 25px;

    margin: 10px 0;

    overflow: visible;
}


.chart-card h3 {

    color: #111827 !important;

    text-align: center;

    font-size: 22px;

    margin-top: 0;

    margin-bottom: 12px;
}


.chart-note {

    color: #4b5563 !important;

    text-align: center;

    font-size: 14px;
}


/* ============================================================
   BMI NUMBER
   ============================================================ */

.bmi-number {

    text-align: center;

    color: #111827 !important;

    font-size: 46px;

    font-weight: 800;

    margin-top: 10px;
}


.bmi-category {

    text-align: center;

    color: #2563eb !important;

    font-size: 20px;

    font-weight: 700;

    margin-bottom: 25px;
}


/* ============================================================
   BMI SCALE
   ============================================================ */

.bmi-scale {

    display: flex;

    width: 100%;

    height: 48px;

    border-radius: 8px;

    overflow: hidden;

    border: 1px solid #9ca3af;
}


.bmi-part {

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 11px;

    font-weight: 700;

    color: #111827 !important;

    text-align: center;
}


.underweight {

    width: 21.25%;

    background: #bfdbfe !important;
}


.normal {

    width: 16.25%;

    background: #bbf7d0 !important;
}


.overweight {

    width: 12.5%;

    background: #fde68a !important;
}


.obesity {

    width: 50%;

    background: #fecaca !important;
}


/* ============================================================
   BMI MARKER
   ============================================================ */

.bmi-marker-container {

    position: relative;

    width: 100%;

    height: 25px;
}


.bmi-marker {

    position: absolute;

    top: -2px;

    transform: translateX(-50%);

    color: #111827 !important;

    font-size: 20px;

    font-weight: 900;
}


.bmi-values {

    display: flex;

    justify-content: space-between;

    color: #374151 !important;

    font-size: 12px;

    margin-top: 0;
}


/* ============================================================
   PROBABILITY CHART
   ============================================================ */

.probability-card {

    min-height: 420px;
}


.probability-row {

    width: 100%;

    margin: 17px 0;
}


.probability-header {

    display: flex;

    justify-content: space-between;

    gap: 10px;

    margin-bottom: 6px;

    color: #111827 !important;

    font-size: 14px;

    font-weight: 600;
}


.probability-track {

    width: 100%;

    height: 24px;

    background: #e5e7eb !important;

    border-radius: 6px;

    overflow: hidden;

    border: 1px solid #d1d5db;
}


.probability-bar {

    height: 100%;

    background: #2563eb !important;

    border-radius: 5px;

    min-width: 0;
}


/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer-card {

    margin-top: 30px;

    margin-bottom: 20px;
}


/* ============================================================
   BUTTON
   ============================================================ */

button {

    font-weight: 700 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .gradio-container {

        padding: 12px !important;
    }


    .app-title h1 {

        font-size: 26px;
    }


    .app-title p {

        font-size: 16px;
    }


    .app-info-card,
    .quick-reminder,
    .disclaimer-card,
    .chart-card {

        padding: 16px;
    }


    .bmi-part {

        font-size: 8px;
    }


    .probability-header {

        font-size: 12px;
    }

}

"""


# ============================================================
# 18. GRADIO APPLICATION
# ============================================================

with gr.Blocks(

    title=(
        "AI/ML-Based Obesity Level Prediction "
        "and Personalized Habit Coaching System"
    ),

    css=CSS,

    theme=gr.themes.Soft()

) as demo:


    # ========================================================
    # TITLE
    # ========================================================

    gr.HTML("""

    <div class="app-title">

        <h1>
            🧠 AI/ML-Based Obesity Level Prediction
            and Personalized Habit Coaching System
        </h1>

        <p>
            Predict obesity level and receive personalized
            lifestyle suggestions
        </p>

    </div>

    """)


    # ========================================================
    # ABOUT APPLICATION
    # ========================================================

    gr.Markdown(
        "## ℹ️ About This Application"
    )


    gr.HTML("""

    <div class="app-info-card">

        <h3>
            What does this application do?
        </h3>

        <p>
            This application uses a machine-learning model
            to estimate an obesity category from physical
            and lifestyle-related information.
        </p>

        <p>
            It also calculates BMI separately and provides
            probability information, simple habit suggestions
            and a personalized action plan.
        </p>

        <p>
            <b>Machine Learning Model:</b>
            Random Forest Classifier with 300 decision trees.
        </p>

        <p>
            <b>Dataset:</b>
            UCI Obesity Dataset.
        </p>

        <p>
            <b>Learning Type:</b>
            Supervised Machine Learning – Multiclass Classification.
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

    <div class="app-info-card">

        <h3>
            🔢 Understanding the Numerical Scales
        </h3>

        <p>
            Some questions use numerical scales instead of
            direct measurements. These values represent the
            coding used by the dataset.
        </p>


        <hr>


        <h4>
            FCVC – Vegetable Consumption Frequency
        </h4>

        <ul>

            <li>
                <b>1</b> → Rarely consume vegetables
            </li>

            <li>
                <b>2</b> → Sometimes / moderate consumption
            </li>

            <li>
                <b>3</b> → Frequently consume vegetables
            </li>

        </ul>

        <p>
            Example: <b>2.3</b> represents a value between
            2 and 3.
        </p>


        <hr>


        <h4>
            NCP – Number of Main Meals
        </h4>

        <ul>

            <li>
                <b>1</b> → About one main meal
            </li>

            <li>
                <b>2</b> → About two main meals
            </li>

            <li>
                <b>3</b> → About three main meals
            </li>

            <li>
                <b>4</b> → About four main meals
            </li>

        </ul>


        <hr>


        <h4>
            CH2O – Water Consumption
        </h4>

        <ul>

            <li>
                <b>1</b> → Low
            </li>

            <li>
                <b>2</b> → Moderate
            </li>

            <li>
                <b>3</b> → High
            </li>

        </ul>

        <p>
            <b>Important:</b>
            This is a dataset scale, not litres of water.
        </p>


        <hr>


        <h4>
            FAF – Physical Activity Frequency
        </h4>

        <ul>

            <li>
                <b>0</b> → Little or no activity
            </li>

            <li>
                <b>1</b> → Low activity
            </li>

            <li>
                <b>2</b> → Moderate activity
            </li>

            <li>
                <b>3</b> → High activity
            </li>

        </ul>


        <hr>


        <h4>
            TUE – Technology Usage Time
        </h4>

        <ul>

            <li>
                <b>0</b> → Low
            </li>

            <li>
                <b>1</b> → Moderate
            </li>

            <li>
                <b>2</b> → High
            </li>

        </ul>

        <p>
            This is a dataset scale and not an exact number
            of hours.
        </p>


        <hr>


        <h4>
            Physical Measurements
        </h4>

        <p>
            <b>Age:</b>
            Enter your age in years.
        </p>

        <p>
            <b>Height:</b>
            Enter height in metres.
            Example: 170 cm = <b>1.70 m</b>.
        </p>

        <p>
            <b>Weight:</b>
            Enter weight in kilograms.
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

            value="Male",

            label="Gender",

            info="Select your gender."

        )


        age = gr.Number(

            value=22,

            label="Age (years)",

            info="Enter your age in years."

        )


        height = gr.Number(

            value=1.70,

            label="Height (metres)",

            info="Example: 170 cm = 1.70 m."

        )


        weight = gr.Number(

            value=70,

            label="Weight (kg)",

            info="Enter weight in kilograms."

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

            value="yes",

            label="Family History with Overweight",

            info="Has anyone in your family had overweight/obesity?"

        )


        favc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            value="No",

            label="Frequent High-Calorie Food",

            info="Do you frequently eat high-calorie foods?"

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

            info="Approximately how many main meals per day?"

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

            label="Food Between Meals",

            info="How often do you eat between main meals?"

        )


        smoke = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            value="No",

            label="Smoking",

            info="Do you currently smoke?"

        )


        ch2o = gr.Slider(

            minimum=1,

            maximum=3,

            value=2,

            step=0.1,

            label="Daily Water Consumption",

            info="1 = low, 2 = moderate, 3 = high. Dataset scale."

        )


        scc = gr.Radio(

            choices=[
                "Yes",
                "No"
            ],

            value="No",

            label="Monitor Calorie Consumption",

            info="Do you monitor your calorie intake?"

        )


    # ========================================================
    # PHYSICAL ACTIVITY & OTHER HABITS
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

            info="0 = low, 1 = moderate, 2 = high."

        )


        calc = gr.Dropdown(

            choices=[
                "No",
                "Sometimes",
                "Frequently",
                "Always"
            ],

            value="Sometimes",

            label="Alcohol Consumption",

            info="How frequently do you consume alcohol?"

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

            label="Transportation",

            info="Select your usual transportation."

        )


    # ========================================================
    # QUICK REMINDER
    # ========================================================

    gr.HTML("""

    <div class="quick-reminder">

        <h3>
            ✅ Quick Input Reminder
        </h3>

        <ul>

            <li>
                Height → metres, e.g. <b>1.70</b>
            </li>

            <li>
                Weight → kilograms, e.g. <b>70</b>
            </li>

            <li>
                FCVC → <b>1–3 scale</b>
            </li>

            <li>
                NCP → approximately <b>1–4 main meals</b>
            </li>

            <li>
                CH2O → <b>1–3 dataset scale</b>, not litres
            </li>

            <li>
                FAF → <b>0–3 scale</b>
            </li>

            <li>
                TUE → <b>0–2 scale</b>
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
    # PREDICTION RESULTS
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


    # ========================================================
    # CHROME-SAFE VISUALS
    # ========================================================

    with gr.Row():

        bmi_result = gr.HTML("""

        <div class="chart-card">

            <h3>
                BMI Indicator
            </h3>

            <p class="chart-note">
                Your BMI indicator will appear here.
            </p>

        </div>

        """)


        probability_result = gr.HTML("""

        <div class="chart-card">

            <h3>
                Prediction Probability
            </h3>

            <p class="chart-note">
                Prediction probabilities will appear here.
            </p>

        </div>

        """)


    # ========================================================
    # UNDERSTANDING RESULTS
    # ========================================================

    gr.HTML("""

    <div class="app-info-card">

        <h3>
            📌 Understanding Your Results
        </h3>

        <p>
            <b>Predicted Obesity Level:</b>
            This is the category predicted by the Random Forest
            machine-learning model.
        </p>

        <p>
            <b>BMI:</b>
            BMI is calculated separately using height and weight.
            It is not directly used as an input feature in the
            current machine-learning model.
        </p>

        <p>
            <b>Prediction Probability:</b>
            The probability section shows the model's estimated
            probability for each of the seven possible categories.
        </p>

        <p>
            <b>Estimated Risk Indicator:</b>
            This combines the model probabilities of the
            non-healthy categories.
        </p>

        <p>
            This risk indicator is not a clinically validated
            medical risk score.
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

    <div class="app-info-card">

        <h3>
            Machine Learning Details
        </h3>

        <ul>

            <li>
                <b>Learning Type:</b>
                Supervised Learning
            </li>

            <li>
                <b>Task:</b>
                Multiclass Classification
            </li>

            <li>
                <b>Algorithm:</b>
                Random Forest Classifier
            </li>

            <li>
                <b>Number of Trees:</b>
                300
            </li>

            <li>
                <b>Train/Test Split:</b>
                80% / 20%
            </li>

            <li>
                <b>Categorical Data:</b>
                One-Hot Encoded
            </li>

            <li>
                <b>Output:</b>
                Seven obesity-related categories
            </li>

        </ul>

        <p>
            The Habit Coach and Action Plan are
            <b>rule-based components</b>.
            They are not generated by the Random Forest model.
        </p>

    </div>

    """)


    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.HTML("""

    <div class="disclaimer-card">

        <h2>
            ⚠️ Disclaimer
        </h2>

        <p>
            This application is developed for
            <b>educational, demonstration and informational
            purposes only</b>.
        </p>

        <p>
            The predictions generated by this application are
            based on a machine-learning model trained on the
            UCI Obesity Dataset and should <b>not</b> be considered
            a medical diagnosis, clinical assessment or
            professional medical advice.
        </p>

        <p>
            BMI and model probabilities have limitations and
            may not accurately represent an individual's complete
            health condition.
        </p>

        <p>
            The Habit Coach and Action Plan provide general
            lifestyle suggestions and are not a substitute for
            advice from a qualified doctor, dietitian,
            nutritionist or other healthcare professional.
        </p>

        <p>
            If you have concerns about your weight, nutrition,
            physical activity or overall health, consult a
            qualified healthcare professional.
        </p>

    </div>

    """)


    # ========================================================
    # CONNECT PREDICT BUTTON
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
            bmi_result,
            probability_result,
            habit_output,
            action_output

        ]

    )


# ============================================================
# 19. LAUNCH
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
