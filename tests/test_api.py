import torch
from fastapi.testclient import TestClient
import pytest
from datetime import datetime
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import json
from fastapi import HTTPException

# Add src to path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

# Mock the dependencies
mock_model = Mock()
mock_model.return_value = Mock(
    eval=Mock(return_value=None),
    __call__=Mock(return_value=(torch.randn(1, 2), {'drug_features': None, 'ehr_features': None}))
)

mock_dose_simulator = Mock()
mock_dose_simulator.calculate_pharmacokinetics.return_value = {'clearance': 1.0}
mock_dose_simulator.simulate_response.return_value = {'dose_range': [0.1, 1.0]}

mock_report_generator = Mock()
mock_report_generator.generate_report.return_value = {
    'response': Mock(
        confidence_interval=(0.4, 0.6),
        recommended_dose=100,
        adverse_effects=['nausea']
    ),
    'recommendations': ['Monitor blood pressure']
}

# Patch the imports
patch('src.api.main.FusionModel', mock_model).start()
patch('src.api.main.DoseSimulator', lambda x: mock_dose_simulator).start()
patch('src.api.main.ClinicalReportGenerator', lambda: mock_report_generator).start()

from src.api.main import app, oauth2_scheme

def get_test_token():
    return "test_token"

# Override token dependency
app.dependency_overrides[oauth2_scheme] = get_test_token

# Initialize test client
client = TestClient(app)

@pytest.fixture
def valid_prediction_request():
    return {
        "patient_id": "TEST001",
        "drug_id": "DRUG001",
        "genomic_data": {
            "gene_expression": {"GENE1": 0.5, "GENE2": -0.3},
            "mutations": {"GENE1": "wild_type", "GENE2": "mutant"},
            "cnv_data": {"GENE1": 2.0, "GENE2": 1.0}
        },
        "clinical_data": {
            "age": 45,
            "weight": 70.5,
            "height": 175.0,
            "sex": "M",
            "comorbidities": ["hypertension"],
            "medications": ["metformin"]
        },
        "ehr_data": {
            "medical_history": [
                {"condition": "type_2_diabetes", "date": "2020-01-01"}
            ],
            "lab_results": {
                "glucose": [120.0, 118.0],
                "hba1c": [6.5, 6.3]
            },
            "vital_signs": {
                "blood_pressure": [120.0, 80.0],
                "heart_rate": [72.0]
            },
            "timestamps": [
                datetime.now().isoformat(),
                (datetime.now()).isoformat()
            ]
        }
    }

@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {get_test_token()}"}

def test_predict_endpoint(valid_prediction_request, auth_headers):
    response = client.post(
        "/predict",
        json=valid_prediction_request,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "patient_id" in data
    assert "response_probability" in data
    assert 0 <= data["response_probability"] <= 1
    assert len(data["confidence_interval"]) == 2
    assert isinstance(data["recommended_dose"], (int, float))
    assert isinstance(data["adverse_effects_risk"], dict)
    assert isinstance(data["recommendations"], list)

def test_batch_predict_endpoint(valid_prediction_request, auth_headers):
    batch_request = {
        "requests": [valid_prediction_request] * 2,
        "priority": 1
    }
    
    # Test batch submission
    response = client.post(
        "/predict/batch",
        json=batch_request,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "processing"
    
    # Test batch status check
    job_id = data["job_id"]
    response = client.get(
        f"/predict/batch/{job_id}",
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] in ["processing", "completed", "failed"]

def test_invalid_request(auth_headers):
    invalid_request = {
        "patient_id": "TEST001",
        "drug_id": "DRUG001",
        # Missing required fields
    }
    response = client.post(
        "/predict",
        json=invalid_request,
        headers=auth_headers
    )
    assert response.status_code == 422  # Validation error

def test_missing_auth():
    with pytest.raises(HTTPException) as exc_info:
        client.post(
            "/predict",
            json={}
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Not authenticated"

def test_invalid_auth():
    with pytest.raises(HTTPException) as exc_info:
        client.post(
            "/predict",
            json={},
            headers={"Authorization": "Bearer invalid_token"}
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid authentication credentials"

# Clean up patches after tests
def teardown_module(module):
    patch.stopall()