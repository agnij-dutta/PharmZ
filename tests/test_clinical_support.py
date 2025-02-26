import unittest
from datetime import datetime
from src.models.clinical_support import DoseSimulator, ClinicalReportGenerator

class TestDoseSimulator(unittest.TestCase):
    def setUp(self):
        self.pk_params = {
            'EC50': 0.5,
            'hill_coefficient': 1.2,
            'base_clearance': 1.0,
            'base_volume': 1.0
        }
        self.simulator = DoseSimulator(self.pk_params)
        
    def test_pharmacokinetics_calculation(self):
        patient_features = {
            'weight': 70,
            'age': 50,
            'height': 170
        }
        
        pk_data = self.simulator.calculate_pharmacokinetics(
            'test_drug',
            patient_features
        )
        
        self.assertIn('clearance', pk_data)
        self.assertIn('volume', pk_data)
        self.assertIn('half_life', pk_data)
        self.assertIn('auc', pk_data)
        self.assertTrue(all(v > 0 for v in pk_data.values()))
        
    def test_response_simulation(self):
        base_response = 0.7
        pk_params = {
            'clearance': 1.0,
            'volume': 1.0,
            'half_life': 12.0
        }
        
        responses = self.simulator.simulate_response(base_response, pk_params)
        
        self.assertEqual(len(responses), len(self.simulator.dose_levels))
        self.assertTrue(all(0 <= r <= 1 for r in responses.values()))
        
class TestClinicalReportGenerator(unittest.TestCase):
    def setUp(self):
        self.report_generator = ClinicalReportGenerator()
        
    def test_report_generation(self):
        predictions = {
            'drug_id': 'test_drug',
            'response': 0.8,
            'uncertainty': 0.1,
            'dose_responses': {2.5: 0.5, 5.0: 0.7, 7.5: 0.8}
        }
        
        pk_data = {
            'clearance': 1.0,
            'volume': 1.0,
            'auc': 1.2,
            'half_life': 12.0
        }
        
        report = self.report_generator.generate_report(
            'test_patient',
            predictions,
            pk_data
        )
        
        self.assertIn('patient_id', report)
        self.assertIn('response', report)
        self.assertIn('recommendations', report)
        self.assertIn('confidence_level', report)
        self.assertIn('monitoring_plan', report)
        
        response = report['response']
        self.assertEqual(response.drug_id, 'test_drug')
        self.assertIsInstance(response.confidence_interval, tuple)
        self.assertIsInstance(response.timestamp, datetime)
        
    def test_confidence_interval_calculation(self):
        point_estimate = 0.7
        uncertainty = 0.1
        
        ci = self.report_generator._calculate_confidence_interval(
            point_estimate,
            uncertainty
        )
        
        self.assertIsInstance(ci, tuple)
        self.assertEqual(len(ci), 2)
        self.assertTrue(ci[0] < point_estimate < ci[1])
        
    def test_monitoring_plan_generation(self):
        from src.models.clinical_support import DrugResponse
        
        response = DrugResponse(
            drug_id='test_drug',
            predicted_response=0.8,
            confidence_interval=(0.7, 0.9),
            recommended_dose=5.0,
            adverse_effects=['High exposure risk'],
            timestamp=datetime.now()
        )
        
        plan = self.report_generator._create_monitoring_plan(response)
        
        self.assertIsInstance(plan, list)
        self.assertTrue(any('adverse effect' in item.lower() for item in plan))
        self.assertTrue(any('baseline' in item.lower() for item in plan))

if __name__ == '__main__':
    unittest.main()