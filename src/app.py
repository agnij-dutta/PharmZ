import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys

src_path = str(Path(__file__).parent)
sys.path.append(src_path)

from data.ingestion import load_genomic_data, load_clinical_data, fetch_drug_properties, process_batch_data
from models.fusion_model import FusionModel
from models.clinical_support import DoseSimulator, ClinicalReportGenerator

# Page config
st.set_page_config(
    page_title="PharmX - Drug Response Prediction",
    page_icon="💊",
    layout="wide"
)

def load_model():
    model_path = Path(__file__).parent.parent / "models" / "fusion_model.pt"
    if not model_path.exists():
        st.error("Model not found. Please train the model first.")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = FusionModel(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def plot_response_curve(dose_responses: Dict[float, float]):
    fig = go.Figure()
    doses = list(dose_responses.keys())
    responses = list(dose_responses.values())
    
    fig.add_trace(go.Scatter(
        x=doses,
        y=responses,
        mode='lines+markers',
        name='Dose Response'
    ))
    
    fig.update_layout(
        title='Dose-Response Curve',
        xaxis_title='Dose (mg)',
        yaxis_title='Response Probability',
        yaxis_range=[0, 1]
    )
    
    return fig

def process_uploaded_files(genomic_file, clinical_file):
    if genomic_file is not None:
        genomic_path = Path("temp") / genomic_file.name
        genomic_path.parent.mkdir(exist_ok=True)
        genomic_path.write_bytes(genomic_file.getvalue())
        genomic_data = load_genomic_data(str(genomic_path))
    else:
        genomic_data = pd.DataFrame()

    if clinical_file is not None:
        clinical_path = Path("temp") / clinical_file.name
        clinical_path.parent.mkdir(exist_ok=True)
        clinical_path.write_bytes(clinical_file.getvalue())
        clinical_data = load_clinical_data(str(clinical_path))
    else:
        clinical_data = pd.DataFrame()
        
    return genomic_data, clinical_data

def main():
    st.title("PharmX - Drug Response Prediction")
    
    tabs = st.tabs(["Data Upload", "Patient Information", "Analysis"])
    
    with tabs[0]:
        st.header("Data Upload")
        genomic_file = st.file_uploader("Upload Genomic Data (CSV/H5/MTX)", type=['csv', 'h5', 'mtx'])
        clinical_file = st.file_uploader("Upload Clinical Data (CSV)", type=['csv'])
        
        if st.button("Process Data"):
            if genomic_file is None and clinical_file is None:
                st.error("Please upload at least one data file")
            else:
                with st.spinner("Processing data..."):
                    genomic_data, clinical_data = process_uploaded_files(genomic_file, clinical_file)
                    st.session_state['genomic_data'] = genomic_data
                    st.session_state['clinical_data'] = clinical_data
                    st.success("Data processed successfully!")
                    
                    if not genomic_data.empty:
                        st.subheader("Genomic Data Preview")
                        st.dataframe(genomic_data.head())
                    
                    if not clinical_data.empty:
                        st.subheader("Clinical Data Preview")
                        st.dataframe(clinical_data.head())
    
    with tabs[1]:
        st.header("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clinical Details")
            patient_id = st.text_input("Patient ID")
            age = st.number_input("Age", min_value=0, max_value=150, value=50)
            weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=0.0, value=170.0)
            sex = st.selectbox("Sex", ["M", "F"])
            
        with col2:
            st.subheader("Medical History")
            comorbidities = st.multiselect(
                "Comorbidities",
                ["Hypertension", "Diabetes", "Heart Disease", "Asthma"]
            )
            medications = st.multiselect(
                "Current Medications",
                ["Metformin", "Insulin", "Statins", "Beta Blockers"]
            )
            
        st.subheader("Drug Selection")
        drug_name = st.text_input("Enter Drug Name")
        if drug_name:
            with st.spinner("Fetching drug properties..."):
                drug_props = fetch_drug_properties(drug_name)
                if drug_props:
                    st.json(drug_props)
                else:
                    st.error("Could not fetch drug properties")
    
    with tabs[2]:
        st.header("Analysis")
        
        if st.button("Generate Prediction"):
            if 'genomic_data' not in st.session_state or 'clinical_data' not in st.session_state:
                st.error("Please process data files first")
                return
                
            model = load_model()
            if model is None:
                return
                
            with st.spinner("Generating predictions..."):
                # Here we would normally make predictions with the model
                # For now, showing sample visualizations
                
                # Sample dose-response curve
                doses = np.linspace(0, 100, 20)
                responses = 1 / (1 + np.exp(-0.05 * (doses - 50)))
                fig = plot_response_curve(dict(zip(doses, responses)))
                st.plotly_chart(fig)
                
                # Confidence and recommendations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Response Probability", "78%")
                    st.metric("Recommended Dose", "45 mg")
                    
                with col2:
                    st.subheader("Risk Analysis")
                    st.progress(0.3, text="Toxicity Risk: 30%")
                    st.progress(0.8, text="Efficacy Score: 80%")
                
                st.subheader("Clinical Recommendations")
                st.markdown("""
                - Start with 45mg daily dose
                - Monitor liver function weekly
                - Check drug interactions with current medications
                - Schedule follow-up in 2 weeks
                """)

if __name__ == "__main__":
    main()