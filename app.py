import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(
    page_title="Concrete AI Assistant",
    page_icon="🏗️",
    layout="wide",
)

MODEL_FILE = Path(__file__).parent / "random_forest_concrete_strength_model.pkl"

FEATURES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

FEATURE_LABELS = {
    "cement": "Cement",
    "blast_furnace_slag": "Blast Furnace Slag",
    "fly_ash": "Fly Ash",
    "water": "Water",
    "superplasticizer": "Superplasticizer",
    "coarse_aggregate": "Coarse Aggregate",
    "fine_aggregate": "Fine Aggregate",
    "age": "Curing Age",
}

UNITS = {
    "cement": "kg/m³",
    "blast_furnace_slag": "kg/m³",
    "fly_ash": "kg/m³",
    "water": "kg/m³",
    "superplasticizer": "kg/m³",
    "coarse_aggregate": "kg/m³",
    "fine_aggregate": "kg/m³",
    "age": "days",
}

DEFAULTS = {
    "cement": 300.0,
    "blast_furnace_slag": 100.0,
    "fly_ash": 0.0,
    "water": 180.0,
    "superplasticizer": 5.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 750.0,
    "age": 28,
}


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_FILE.name}. Keep it in the same folder as app.py."
        )
    return joblib.load(MODEL_FILE)


def predict_strength(model, values):
    """Predict one mix using the feature names expected by the trained model."""
    row = {feature: values[feature] for feature in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES)
    prediction = float(model.predict(frame)[0])
    return prediction


def get_inputs(prefix="main", defaults=None):
    """Render the eight model inputs in two columns."""
    defaults = defaults or DEFAULTS
    values = {}
    col1, col2 = st.columns(2)

    with col1:
        for feature in FEATURES[:4]:
            values[feature] = st.number_input(
                f"{FEATURE_LABELS[feature]} ({UNITS[feature]})",
                min_value=0.0,
                value=float(defaults[feature]),
                step=1.0,
                key=f"{prefix}_{feature}",
                help="Enter the quantity per cubic metre of concrete.",
            )

    with col2:
        for feature in FEATURES[4:]:
            min_value = 1.0 if feature == "age" else 0.0
            values[feature] = st.number_input(
                f"{FEATURE_LABELS[feature]} ({UNITS[feature]})",
                min_value=min_value,
                value=float(defaults[feature]),
                step=1.0,
                key=f"{prefix}_{feature}",
                help=(
                    "Curing age must be at least one day."
                    if feature == "age"
                    else "Enter the quantity per cubic metre of concrete."
                ),
            )

    values["age"] = int(values["age"])
    return values


def feature_importance_frame(model):
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(model.feature_importances_, dtype=float)
    if len(values) != len(FEATURES):
        return None
    frame = pd.DataFrame({
        "Feature": [FEATURE_LABELS[f] for f in FEATURES],
        "Importance": values,
        "Model feature": FEATURES,
    })
    return frame.sort_values("Importance", ascending=False)


def mix_frame(values):
    return pd.DataFrame([{f: values[f] for f in FEATURES}], columns=FEATURES)


def render_result_explanation(prediction):
    st.markdown("#### How to interpret this estimate")
    st.write(
        f"The model estimates a compressive strength of **{prediction:.2f} MPa** "
        "for the entered mixture and curing age. This is a prediction from the "
        "trained Random Forest model, not a laboratory test result."
    )
    st.caption(
        "Use this result for educational and preliminary comparison only. "
        "Validate concrete performance through appropriate laboratory testing "
        "and follow applicable design codes and professional engineering judgment."
    )


# ----------------------------
# Navigation and shared model
# ----------------------------
st.sidebar.title("🏗️ Concrete AI Assistant")
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Strength Predictor",
        "Explainable AI",
        "Mix Comparison",
        "Engineering Assistant",
        "Dataset and Model",
        "Program Relevance",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Civil Engineering AI Project")

try:
    model = load_model()
except Exception as error:
    model = None
    st.sidebar.error("Model could not be loaded.")
    st.sidebar.caption(str(error))

# ----------------------------
# Home
# ----------------------------
if page == "Home":
    st.title("Concrete AI Assistant")
    st.subheader("AI-Based Concrete Compressive Strength Prediction")
    st.write(
        "Estimate concrete compressive strength using a trained Random Forest "
        "Regression model, inspect model-level feature importance, compare "
        "alternative mixtures, and ask general concrete-engineering questions."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction Model", "Random Forest")
    c2.metric("Input Features", "8")
    c3.metric("Output", "Strength (MPa)")

    st.markdown("### Available tools")
    st.markdown(
        "- **Strength Predictor:** estimate strength for one concrete mixture.\n"
        "- **Explainable AI:** review the model's overall feature-importance scores.\n"
        "- **Mix Comparison:** compare predictions for up to five alternative mixtures.\n"
        "- **Engineering Assistant:** ask concrete-related questions; optional LLM support "
        "requires an API key."
    )
    st.info(
        "For educational and preliminary estimation only. The app does not replace "
        "laboratory testing, approved mix design procedures, applicable codes, or "
        "professional engineering judgment."
    )

# ----------------------------
# Strength Predictor
# ----------------------------
elif page == "Strength Predictor":
    st.title("Concrete Compressive Strength Predictor")
    st.write("Enter the concrete mixture proportions and curing age.")
    if model is None:
        st.error("The model is unavailable. Check that the .pkl file is beside app.py.")
    else:
        values = get_inputs("predictor")
        if st.button("Predict Strength", type="primary", use_container_width=True):
            try:
                prediction = predict_strength(model, values)
                st.session_state["last_prediction"] = {
                    "inputs": values,
                    "prediction": prediction,
                }
                st.success(f"Estimated Compressive Strength: {prediction:.2f} MPa")
                render_result_explanation(prediction)
                with st.expander("Review entered mixture"):
                    st.dataframe(
                        pd.DataFrame({
                            "Input": [FEATURE_LABELS[f] for f in FEATURES],
                            "Value": [values[f] for f in FEATURES],
                            "Unit": [UNITS[f] for f in FEATURES],
                        }),
                        hide_index=True,
                        use_container_width=True,
                    )
            except Exception as error:
                st.error(f"Prediction failed: {error}")

# ----------------------------
# Explainable AI
# ----------------------------
elif page == "Explainable AI":
    st.title("Explainable AI")
    st.write(
        "This page reports the Random Forest model's built-in feature importance. "
        "These scores describe the model's overall use of each input feature; they "
        "do not prove that a feature causes a strength change in every mixture."
    )

    if model is None:
        st.error("The model is unavailable.")
    else:
        importance = feature_importance_frame(model)
        if importance is None:
            st.warning(
                "This model does not expose a compatible feature_importances_ attribute."
            )
        else:
            top = importance.iloc[0]
            st.metric("Highest model-level importance", top["Feature"], f'{top["Importance"]:.1%}')
            chart_data = importance.set_index("Feature")[["Importance"]]
            st.bar_chart(chart_data, horizontal=True)
            st.dataframe(
                importance[["Feature", "Importance"]].assign(
                    **{"Importance (%)": (importance["Importance"] * 100).round(2)}
                )[["Feature", "Importance (%)"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Importance scores are relative and generally sum to 1. They are not "
                "the percentage of strength caused by each ingredient."
            )

        st.markdown("### Explore a prediction")
        st.write(
            "Enter a mixture to see its prediction and compare its inputs with the "
            "model's overall feature-importance ranking."
        )
        values = get_inputs("explain")
        if st.button("Analyze This Mixture", type="primary"):
            try:
                prediction = predict_strength(model, values)
                st.success(f"Predicted strength: {prediction:.2f} MPa")
                render_result_explanation(prediction)
                importance = feature_importance_frame(model)
                if importance is not None:
                    st.markdown("#### Inputs ordered by overall model importance")
                    explanation = importance.copy()
                    explanation["Entered value"] = explanation["Model feature"].map(values)
                    explanation["Unit"] = explanation["Model feature"].map(UNITS)
                    st.dataframe(
                        explanation[["Feature", "Entered value", "Unit", "Importance"]].assign(
                            Importance=lambda d: (d["Importance"] * 100).round(2).astype(str) + "%"
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.info(
                    "This is a model-level explanation, not a SHAP or causal explanation "
                    "of why this individual prediction has its particular value."
                )
            except Exception as error:
                st.error(f"Analysis failed: {error}")

# ----------------------------
# Mix Comparison
# ----------------------------
elif page == "Mix Comparison":
    st.title("Concrete Mix Comparison")
    st.write(
        "Compare up to five mixture alternatives. The predictions are generated by "
        "the same trained model so that the alternatives can be reviewed consistently."
    )

    if model is None:
        st.error("The model is unavailable.")
    else:
        number_of_mixes = st.selectbox("Number of mixtures to compare", [2, 3, 4, 5], index=1)
        all_values = []
        for i in range(number_of_mixes):
            with st.expander(f"Mixture {i + 1}", expanded=(i < 2)):
                all_values.append(get_inputs(f"mix_{i+1}"))

        if st.button("Compare Mixtures", type="primary", use_container_width=True):
            try:
                rows = []
                for i, values in enumerate(all_values, start=1):
                    rows.append({
                        "Mixture": f"Mixture {i}",
                        "Predicted strength (MPa)": predict_strength(model, values),
                        **{FEATURE_LABELS[f]: values[f] for f in FEATURES},
                    })
                results = pd.DataFrame(rows)
                results = results.sort_values(
                    "Predicted strength (MPa)", ascending=False
                ).reset_index(drop=True)
                st.markdown("### Results")
                best = results.iloc[0]
                st.metric(
                    "Highest predicted strength",
                    f'{best["Predicted strength (MPa)"]:.2f} MPa',
                    best["Mixture"],
                )
                st.bar_chart(
                    results.set_index("Mixture")[["Predicted strength (MPa)"]],
                    horizontal=True,
                )
                st.dataframe(results, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download comparison as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="concrete_mix_comparison.csv",
                    mime="text/csv",
                )
                st.warning(
                    "The highest predicted strength is not automatically the best or "
                    "most economical mix. Workability, durability, exposure, materials, "
                    "cost, specifications, and laboratory validation must also be considered."
                )
            except Exception as error:
                st.error(f"Comparison failed: {error}")

# ----------------------------
# Engineering Assistant
# ----------------------------
elif page == "Engineering Assistant":
    st.title("Concrete Engineering Assistant")
    st.write(
        "Ask general questions about concrete materials, compressive strength, curing, "
        "or how to interpret this app's predictions."
    )

    st.info(
        "Without an API key, this page provides a basic built-in guidance mode. "
        "To enable LLM responses, set OPENAI_API_KEY as an environment variable or "
        "in Streamlit secrets. Never commit the key to source control."
    )

    question = st.text_area(
        "Your question",
        placeholder="Example: Why does curing age affect concrete compressive strength?",
        height=120,
    )

    if st.button("Ask Assistant", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY", "")
                except Exception:
                    api_key = ""

            if api_key:
                try:
                    from openai import OpenAI

                    client = OpenAI(api_key=api_key)
                    with st.spinner("Preparing an answer..."):
                        response = client.responses.create(
                            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                            instructions=(
                                "You are an educational concrete-engineering assistant. "
                                "Give clear, cautious explanations. Do not present AI "
                                "predictions as certified design values. Mention that "
                                "mix design and structural decisions require applicable "
                                "codes, laboratory testing, and qualified professionals."
                            ),
                            input=question.strip(),
                        )
                    st.markdown(response.output_text)
                except Exception as error:
                    st.error(
                        "The LLM request failed. Check the API key, model access, "
                        f"network connection, and package installation. Details: {error}"
                    )
            else:
                st.markdown("### General guidance")
                st.write(
                    "I can offer only limited built-in guidance while an LLM API key "
                    "is not configured. For this topic, review the relevant concrete "
                    "materials references, your dataset documentation, and the "
                    "limitations of the prediction model."
                )
                st.write(
                    "Question received: " + question.strip()
                )
                st.caption(
                    "Configure OPENAI_API_KEY to enable generated, question-specific answers."
                )

# ----------------------------
# Dataset and Model
# ----------------------------
elif page == "Dataset and Model":
    st.title("Dataset and AI Model")
    st.write(
        "The project is configured around the UCI Concrete Compressive Strength dataset, "
        "which contains eight input variables and a compressive-strength target."
    )

    st.markdown("### Input features")
    st.dataframe(
        pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Model column": FEATURES,
            "Unit": [UNITS[f] for f in FEATURES],
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Algorithm")
    st.write(
        "Random Forest Regression combines predictions from multiple decision trees. "
        "It can model nonlinear relationships among mixture proportions, curing age, "
        "and measured compressive strength."
    )

    st.markdown("### Evaluation metrics")
    st.write(
        "Use the MAE, RMSE, and R² values calculated on a held-out test set or through "
        "cross-validation. Those values are not included in the uploaded model file, "
        "so this page does not invent or display unverified scores."
    )
    st.markdown(
        "[UCI Concrete Compressive Strength dataset]"
        "(https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)"
    )

    if model is not None:
        st.markdown("### Model information")
        st.write(f"Model class: `{type(model).__name__}`")
        st.write(f"Expected input features: `{getattr(model, 'n_features_in_', 'Unknown')}`")
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            st.write("Stored feature names: " + ", ".join(map(str, names)))

# ----------------------------
# Program Relevance
# ----------------------------
elif page == "Program Relevance":
    st.title("Program Relevance")
    st.write(
        "This project connects artificial intelligence with Civil Engineering by "
        "demonstrating how machine-learning regression can estimate concrete "
        "compressive strength from mixture proportions and curing age."
    )
    st.markdown("### Potential educational value")
    st.markdown(
        "- Demonstrates the use of data-driven methods in concrete materials research.\n"
        "- Supports preliminary comparison of candidate mixtures.\n"
        "- Helps students explore relationships between mixture inputs and predicted output.\n"
        "- Encourages discussion of model evaluation, interpretability, and responsible use of AI."
    )
    st.warning(
        "The application is not a substitute for concrete mix-design standards, "
        "laboratory testing, structural design checks, or professional engineering judgment."
    )
