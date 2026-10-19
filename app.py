import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Concrete AI Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = Path(__file__).parent / "random_forest_concrete_strength_model.pkl"
FEATURES = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
]
LABELS = {
    "cement": "Cement",
    "blast_furnace_slag": "Blast Furnace Slag",
    "fly_ash": "Fly Ash",
    "water": "Water",
    "superplasticizer": "Superplasticizer",
    "coarse_aggregate": "Coarse Aggregate",
    "fine_aggregate": "Fine Aggregate",
    "age": "Curing Age",
}
UNITS = {f: ("days" if f == "age" else "kg/m³") for f in FEATURES}
DEFAULTS = {
    "cement": 300.0, "blast_furnace_slag": 100.0, "fly_ash": 0.0,
    "water": 180.0, "superplasticizer": 5.0,
    "coarse_aggregate": 1000.0, "fine_aggregate": 750.0, "age": 28,
}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1250px;}
[data-testid="stMetric"] {background: rgba(120,120,120,0.07); padding: 14px 16px; border-radius: 12px;}
div.stButton > button[kind="primary"] {border-radius: 10px; font-weight: 650; min-height: 2.8rem;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
.hero {padding: 1.4rem 1.5rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 1rem;}
.small-muted {font-size: .9rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE.name}")
    return joblib.load(MODEL_FILE)


def make_input_frame(values):
    return pd.DataFrame([{f: values[f] for f in FEATURES}], columns=FEATURES)


def predict(model, values):
    return float(model.predict(make_input_frame(values))[0])


def input_form(prefix, defaults=None):
    defaults = defaults or DEFAULTS
    values = {}
    left, right = st.columns(2)
    with left:
        for f in FEATURES[:4]:
            values[f] = st.number_input(
                f"{LABELS[f]} ({UNITS[f]})",
                min_value=0.0, value=float(defaults[f]), step=1.0,
                key=f"{prefix}_{f}",
            )
    with right:
        for f in FEATURES[4:]:
            values[f] = st.number_input(
                f"{LABELS[f]} ({UNITS[f]})",
                min_value=1.0 if f == "age" else 0.0,
                value=float(defaults[f]), step=1.0, key=f"{prefix}_{f}",
            )
    values["age"] = int(values["age"])
    return values


def importance_frame(model):
    vals = getattr(model, "feature_importances_", None)
    if vals is None or len(vals) != len(FEATURES):
        return None
    return pd.DataFrame({
        "Feature": [LABELS[f] for f in FEATURES],
        "Model feature": FEATURES,
        "Importance": np.asarray(vals, dtype=float),
    }).sort_values("Importance", ascending=False)


def show_caveat():
    st.caption(
        "Educational and preliminary estimation only. Model predictions are not "
        "laboratory test results and must not replace applicable standards, "
        "validated mix design, laboratory testing, or professional engineering judgment."
    )


def render_shap(model, values):
    try:
        import shap
    except ImportError:
        st.warning("SHAP is not installed. Add `shap` to requirements.txt and redeploy.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(make_input_frame(values))
        sv = np.asarray(shap_values.values)
        if sv.ndim == 3:
            sv = sv[0, :, 0]
        else:
            sv = sv.reshape(-1)
        base_values = np.asarray(shap_values.base_values).reshape(-1)
        base = float(base_values[0]) if len(base_values) else float(np.mean(model.predict(
            pd.DataFrame([DEFAULTS], columns=FEATURES)
        )))
        explain_df = pd.DataFrame({
            "Input": [LABELS[f] for f in FEATURES],
            "Entered value": [values[f] for f in FEATURES],
            "Unit": [UNITS[f] for f in FEATURES],
            "SHAP contribution (MPa)": sv,
            "Absolute contribution": np.abs(sv),
        }).sort_values("Absolute contribution", ascending=False)

        st.markdown("#### Individual prediction explanation")
        st.write(
            "SHAP values show how each input moves this prediction away from the "
            "model's baseline output. Positive values push the prediction higher; "
            "negative values push it lower, relative to that baseline."
        )
        chart = explain_df.set_index("Input")[["SHAP contribution (MPa)"]]
        st.bar_chart(chart, horizontal=True)
        st.dataframe(
            explain_df[["Input", "Entered value", "Unit", "SHAP contribution (MPa)"]]
            .round(4),
            hide_index=True, use_container_width=True,
        )
        st.caption(f"Model baseline output for this explainer: approximately {base:.2f} MPa.")
        st.info(
            "SHAP explains this model prediction, not a causal physical effect. "
            "Interpret contributions in the context of the training dataset."
        )
    except Exception as e:
        st.error(f"Could not calculate SHAP values for this model: {e}")


def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    return key


st.sidebar.title("🏗️ Concrete AI Assistant")
st.sidebar.caption("AI-assisted concrete strength estimation")
page = st.sidebar.radio(
    "Navigate",
    [
        "Home", "Strength Predictor", "SHAP Explainable AI",
        "Model Evaluation", "Mix Comparison", "Engineering Assistant",
        "Dataset and Model", "Program Relevance",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Civil Engineering Research Project")

try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)
    st.sidebar.error("Model unavailable")
    st.sidebar.caption(model_error)

if page == "Home":
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("Concrete AI Assistant")
    st.subheader("AI-Based Concrete Compressive Strength Prediction")
    st.write(
        "A machine-learning web application for predicting concrete compressive "
        "strength from mixture proportions and curing age, explaining predictions, "
        "and comparing candidate mixtures."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Prediction algorithm", "Random Forest")
    b.metric("Input variables", "8")
    c.metric("Prediction unit", "MPa")
    st.markdown("### Explore the application")
    cards = st.columns(2)
    with cards[0]:
        st.markdown("**Strength Predictor**")
        st.write("Estimate compressive strength from an entered concrete mixture.")
        st.markdown("**SHAP Explainable AI**")
        st.write("Inspect how individual inputs contribute to a specific prediction.")
        st.markdown("**Model Evaluation**")
        st.write("Calculate MAE, RMSE, and R² using a labeled evaluation CSV.")
    with cards[1]:
        st.markdown("**Mix Comparison**")
        st.write("Compare alternative mixtures using the same trained model.")
        st.markdown("**Engineering Assistant**")
        st.write("Ask concrete-related questions with optional LLM API support.")
        st.markdown("**Dataset and Model**")
        st.write("Review the model features and methodology.")
    st.info(
        "Model quality metrics are not guessed. Upload a valid labeled test dataset "
        "on the Model Evaluation page to calculate scores."
    )
    show_caveat()

elif page == "Strength Predictor":
    st.title("Concrete Strength Predictor")
    st.write("Enter the concrete mixture proportions and curing age.")
    if model is None:
        st.error(f"Could not load the model: {model_error}")
    else:
        values = input_form("predict")
        if st.button("Predict Compressive Strength", type="primary", use_container_width=True):
            try:
                result = predict(model, values)
                st.session_state["last_values"] = values
                st.session_state["last_prediction"] = result
                st.success("Prediction complete")
                st.metric("Estimated compressive strength", f"{result:.2f} MPa")
                st.markdown("#### Entered mixture")
                st.dataframe(pd.DataFrame({
                    "Feature": [LABELS[f] for f in FEATURES],
                    "Value": [values[f] for f in FEATURES],
                    "Unit": [UNITS[f] for f in FEATURES],
                }), hide_index=True, use_container_width=True)
                show_caveat()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "SHAP Explainable AI":
    st.title("SHAP Explainable AI")
    st.write(
        "SHAP provides a local explanation for an individual model prediction. "
        "This is different from the Random Forest's overall feature-importance ranking."
    )
    if model is None:
        st.error(f"Could not load the model: {model_error}")
    else:
        values = input_form("shap")
        if st.button("Explain This Prediction", type="primary", use_container_width=True):
            try:
                result = predict(model, values)
                st.metric("Predicted compressive strength", f"{result:.2f} MPa")
                render_shap(model, values)
                show_caveat()
            except Exception as e:
                st.error(f"Could not generate prediction explanation: {e}")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.write(
        "Calculate MAE, RMSE, and R² on a labeled test/evaluation dataset. "
        "The uploaded file must contain the same eight input columns as the model "
        "and a measured compressive-strength target column."
    )
    st.warning(
        "Use an independent held-out test set or clearly describe your evaluation "
        "method. Do not report training-set scores as test performance."
    )
    if model is None:
        st.error(f"Could not load the model: {model_error}")
    else:
        uploaded = st.file_uploader("Upload labeled evaluation data (.csv)", type=["csv"])
        target_name = st.text_input(
            "Target column name",
            value="concrete_compressive_strength",
            help="Common UCI dataset target name. Change this to match your CSV exactly.",
        )
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
                st.write("Preview of uploaded dataset")
                st.dataframe(data.head(), use_container_width=True)
                missing = [f for f in FEATURES if f not in data.columns]
                if missing:
                    st.error("Missing required input columns: " + ", ".join(missing))
                elif target_name not in data.columns:
                    st.error(
                        f"Target column '{target_name}' was not found. Available columns: "
                        + ", ".join(map(str, data.columns))
                    )
                else:
                    clean = data[FEATURES + [target_name]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(clean) < 2:
                        st.error("At least two complete numeric rows are required.")
                    elif st.button("Calculate Evaluation Metrics", type="primary"):
                        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                        y_true = clean[target_name].to_numpy(dtype=float)
                        y_pred = model.predict(clean[FEATURES])
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                        r2 = r2_score(y_true, y_pred)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAE", f"{mae:.4f} MPa")
                        m2.metric("RMSE", f"{rmse:.4f} MPa")
                        m3.metric("R²", f"{r2:.4f}")
                        st.write(f"Rows evaluated: **{len(clean)}**")
                        scored = clean.copy()
                        scored["Predicted strength (MPa)"] = y_pred
                        scored["Residual (actual - predicted)"] = y_true - y_pred
                        st.dataframe(scored, hide_index=True, use_container_width=True)
                        st.download_button(
                            "Download scored results",
                            data=scored.to_csv(index=False).encode("utf-8"),
                            file_name="concrete_model_evaluation_results.csv",
                            mime="text/csv",
                        )
                        st.caption(
                            "Metrics reflect only the rows and target values in the uploaded file. "
                            "Record the dataset split and evaluation method in your report."
                        )
            except Exception as e:
                st.error(f"Could not read or evaluate this CSV: {e}")

elif page == "Mix Comparison":
    st.title("Concrete Mix Comparison")
    st.write("Compare predictions for up to five candidate concrete mixtures.")
    if model is None:
        st.error(f"Could not load the model: {model_error}")
    else:
        n = st.selectbox("Number of mixtures", [2, 3, 4, 5], index=1)
        mixtures = []
        for i in range(n):
            with st.expander(f"Mixture {i + 1}", expanded=(i < 2)):
                mixtures.append(input_form(f"mix{i+1}"))
        if st.button("Compare Mixtures", type="primary", use_container_width=True):
            try:
                rows = []
                for i, vals in enumerate(mixtures, start=1):
                    rows.append({
                        "Mixture": f"Mixture {i}",
                        "Predicted strength (MPa)": predict(model, vals),
                        **{LABELS[f]: vals[f] for f in FEATURES},
                    })
                result = pd.DataFrame(rows).sort_values(
                    "Predicted strength (MPa)", ascending=False
                ).reset_index(drop=True)
                best = result.iloc[0]
                st.metric("Highest predicted strength", f'{best["Predicted strength (MPa)"]:.2f} MPa', best["Mixture"])
                st.bar_chart(result.set_index("Mixture")[["Predicted strength (MPa)"]], horizontal=True)
                st.dataframe(result, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download comparison CSV", result.to_csv(index=False).encode("utf-8"),
                    "concrete_mix_comparison.csv", "text/csv"
                )
                st.warning(
                    "The highest predicted strength is not automatically the best mix. "
                    "Workability, durability, cost, exposure conditions, specifications, "
                    "and laboratory validation must also be considered."
                )
            except Exception as e:
                st.error(f"Comparison failed: {e}")

elif page == "Engineering Assistant":
    st.title("Concrete Engineering Assistant")
    st.write(
        "Ask about concrete materials, curing, compressive strength, or interpreting "
        "this application's predictions."
    )
    st.info(
        "Configure OPENAI_API_KEY in Streamlit Secrets to enable AI-generated responses. "
        "Without a key, the page will explain how to enable the feature but cannot call an LLM."
    )
    question = st.text_area(
        "Your question",
        placeholder="Example: How does curing age affect compressive strength?",
        height=120,
    )
    if st.button("Ask Engineering Assistant", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            key = get_api_key()
            if not key:
                st.error(
                    "No API key configured. In Streamlit, open App settings → Secrets and add "
                    'OPENAI_API_KEY = "your-api-key". Do not put the key in app.py or GitHub.'
                )
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=key)
                    with st.spinner("Generating answer..."):
                        response = client.responses.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                            instructions=(
                                "You are an educational concrete-engineering assistant. "
                                "Explain concepts clearly and distinguish general information "
                                "from project-specific design advice. Never certify AI predictions "
                                "as design values. Recommend relevant standards, laboratory "
                                "testing, and qualified engineering review for real projects."
                            ),
                            input=question.strip(),
                        )
                    st.markdown(response.output_text)
                except Exception as e:
                    st.error(
                        "The LLM request failed. Check your API key, API access, selected model, "
                        f"and package installation. Details: {e}"
                    )

elif page == "Dataset and Model":
    st.title("Dataset and Model")
    st.markdown("### Model input features")
    st.dataframe(pd.DataFrame({
        "Feature": [LABELS[f] for f in FEATURES],
        "Column name": FEATURES,
        "Unit": [UNITS[f] for f in FEATURES],
    }), hide_index=True, use_container_width=True)
    st.markdown("### Random Forest Regression")
    st.write(
        "Random Forest Regression combines predictions from multiple decision trees "
        "to estimate a continuous value. Its performance depends on training data, "
        "preprocessing, model settings, and the evaluation method."
    )
    st.markdown(
        "[UCI Concrete Compressive Strength dataset]"
        "(https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)"
    )
    if model is not None:
        st.markdown("### Loaded model details")
        st.write(f"Model type: `{type(model).__name__}`")
        st.write(f"Expected input count: `{getattr(model, 'n_features_in_', 'unknown')}`")
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            st.write("Stored feature names: " + ", ".join(map(str, names)))
        imp = importance_frame(model)
        if imp is not None:
            st.markdown("### Overall feature importance")
            st.bar_chart(imp.set_index("Feature")[["Importance"]], horizontal=True)
    st.caption(
        "Evaluation metrics are intentionally calculated from a supplied labeled test CSV; "
        "they are not fabricated from the model file alone."
    )

elif page == "Program Relevance":
    st.title("Program Relevance")
    st.write(
        "This project demonstrates the application of machine learning to a Civil "
        "Engineering materials problem: estimating concrete compressive strength from "
        "mixture proportions and curing age."
    )
    st.markdown("### Educational applications")
    st.markdown(
        "- Demonstrates a regression model applied to concrete materials data.\n"
        "- Allows preliminary comparison of candidate mixtures.\n"
        "- Shows model-level and individual-prediction explanations.\n"
        "- Supports model evaluation using MAE, RMSE, and R².\n"
        "- Encourages responsible interpretation of AI predictions."
    )
    st.warning(
        "This application does not replace mix-design standards, laboratory testing, "
        "structural design checks, or professional engineering judgment."
    )
