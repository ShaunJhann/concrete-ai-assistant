# Concrete AI Assistant — Final Upgrade

## Included features
- Random Forest compressive-strength prediction
- SHAP explanation for individual predictions
- Model evaluation using MAE, RMSE, and R² from a labeled CSV supplied by the user
- Concrete mix comparison and CSV export
- Optional LLM Engineering Assistant using an API key
- Updated, responsive Streamlit layout

## Files
- `app.py`
- `random_forest_concrete_strength_model.pkl`
- `requirements.txt`

## Deploy
Upload all three files to the root of your GitHub repository. In Streamlit Community Cloud, select `app.py` as the main file.

## Important: evaluation data
The model `.pkl` file alone does not contain the ground-truth test labels needed to verify MAE, RMSE, and R². On the Model Evaluation page, upload a CSV containing these exact input columns:
`cement`, `blast_furnace_slag`, `fly_ash`, `water`, `superplasticizer`, `coarse_aggregate`, `fine_aggregate`, `age`
and a numeric target column (default: `concrete_compressive_strength`).
Use an independent test set for reported test metrics and describe the evaluation method in your research.

## Optional LLM
In Streamlit Community Cloud, open your app settings and configure Secrets:
`OPENAI_API_KEY = "your-api-key"`
Never commit API keys to GitHub. The default model is `gpt-4.1-mini`; set `OPENAI_MODEL` if you want to use another model available to your API account.

## Compatibility
The model was saved using scikit-learn 1.9.0. The requirements pin that version to reduce model-loading compatibility issues. Use a Python runtime supported by the pinned packages.

## Limitations
Predictions and explanations are educational tools. SHAP values describe model behavior, not causal physical effects. Validate real concrete mixtures through suitable laboratory testing and professional engineering judgment.
