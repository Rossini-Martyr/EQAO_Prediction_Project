import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Load the unified pipeline (Preprocessor + Random Forest)
    pipeline = joblib.load('ontario_school_model.pkl')
    
    # Extract components for SHAP calculations
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Create the SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return pipeline, preprocessor, explainer

pipeline, preprocessor, explainer = load_assets()

# --- 2. UI SETUP ---
st.set_page_config(page_title="Ontario School Risk Explorer", layout="wide")
st.title("🏫 School Performance Risk Predictor")
st.markdown("""
This tool uses a machine learning pipeline to identify schools at risk of falling below provincial standards.
It utilizes socio-economic indicators and historical EQAO trends.
""")

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("Input School Characteristics")

enrolment = st.sidebar.number_input("Total Enrolment", 10, 5000, 450)
percent_special_ed = st.sidebar.slider("Special Education Services (%)", 0.0, 100.0, 15.0)
three_year_diff = st.sidebar.slider("3-Year Math Change (%)", -75.0, 75.0, 0.0)
percent_no_degree = st.sidebar.slider("Parents with No Degree/Diploma (%)", 0.0, 100.0, 10.0)
percent_low_income = st.sidebar.slider("Low-Income Households (%)", 0.0, 100.0, 10.0)

# --- 4. DATA PREPARATION ---
# Initialize DataFrame with all columns expected by the pipeline schema
expected_columns = pipeline.feature_names_in_
input_df = pd.DataFrame(columns=expected_columns, index=[0])

# Map UI inputs to specific DataFrame columns
input_df['Enrolment'] = enrolment
input_df['3 Year Diff'] = three_year_diff
input_df['Percentage of Students Receiving Special Education Services'] = percent_special_ed
input_df['Percentage of Students Whose Parents Have No Degree, Diploma or Certificate'] = percent_no_degree
input_df['Percentage of School-Aged Children Who Live in Low-Income Households'] = percent_low_income

# Note: Other columns remain as np.nan and will be filled by the pipeline's Imputer

# --- 5. PREDICTION ---
if st.button("Generate Risk Analysis"):
    # Generate Probability
    prob = pipeline.predict_proba(input_df)[0, 1]
    
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Result")
        st.metric("Risk Probability", f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("Classification: High Risk")
        else:
            st.success("Classification: Low Risk / Stable")

    # --- 6. NATURAL LANGUAGE INTERPRETATION (SHAP DRIVERS) ---
    with col2:
        st.subheader("Key Risk Drivers")
        
        # Transform data and calculate SHAP
        X_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        shap_values = explainer.shap_values(X_transformed)

        # Handle SHAP versioning differences (List vs Array)
        if isinstance(shap_values, list):
            display_shap = shap_values[1][0] # Class 1, First Row
        else:
            # Handle 3D arrays if present, otherwise 2D
            display_shap = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

        # Create impact summary
        impact_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact': display_shap
        })

        # Logic to extract drivers
        top_pos = impact_df[impact_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(2)
        top_neg = impact_df[impact_df['Impact'] < 0].sort_values(by='Impact', ascending=True).head(2)

        st.write(f"The model's probability score of **{prob*100:.1f}%** is primarily driven by the following factors:")

        # Output to UI
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.write("🔼 **Factors Increasing Risk**")
            if not top_pos.empty:
                for _, row in top_pos.iterrows():
                    clean_name = row['Feature'].split('__')[-1]
                    st.write(f"- {clean_name}")
            else:
                st.write("- None detected")

        with res_col2:
            st.write("🔽 **Factors Decreasing Risk**")
            if not top_neg.empty:
                for _, row in top_neg.iterrows():
                    clean_name = row['Feature'].split('__')[-1]
                    st.write(f"- {clean_name}")
            else:
                st.write("- None detected")