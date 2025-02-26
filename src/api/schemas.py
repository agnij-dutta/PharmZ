from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class GenomicData(BaseModel):
    gene_expression: Dict[str, float]
    mutations: Dict[str, str]
    cnv_data: Optional[Dict[str, float]] = None

class ClinicalData(BaseModel):
    age: int = Field(..., ge=0, le=150)
    weight: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    sex: str = Field(..., pattern="^(M|F)$")
    comorbidities: List[str] = []
    medications: List[str] = []

class EHRData(BaseModel):
    medical_history: List[Dict[str, str]]
    lab_results: Dict[str, List[float]]
    vital_signs: Dict[str, List[float]]
    timestamps: List[datetime]

class PredictionRequest(BaseModel):
    patient_id: str
    drug_id: str
    genomic_data: GenomicData
    clinical_data: ClinicalData
    ehr_data: Optional[EHRData] = None

class PredictionResponse(BaseModel):
    patient_id: str
    drug_id: str
    response_probability: float = Field(..., ge=0, le=1)
    confidence_interval: tuple[float, float]
    recommended_dose: float
    adverse_effects_risk: Dict[str, float]
    next_monitoring_date: datetime
    recommendations: List[str]

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]
    priority: Optional[int] = Field(1, ge=1, le=5)

class BatchPredictionResponse(BaseModel):
    job_id: str
    status: str
    estimated_completion_time: datetime
    predictions: Optional[List[PredictionResponse]] = None