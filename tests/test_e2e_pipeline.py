# -*- coding: utf-8 -*-
"""
End-to-End Pipeline Validation Suite (Phase 6: Step 6.1).

Validates:
1. Robustness against unseen localities, cities, and missing fields.
2. Extreme listing boundaries (micro-units to luxury estates, vintage to futuristic).
3. Rapid continuous queries & latency SLA benchmarks (p50, p95).
4. Real-time remote MLflow audit logging & telemetry integrity on DagsHub.
"""

from typing import Dict, Any, List
import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Reconfigure stdout for utf-8
reconfig = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig):
    reconfig(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from src.mlflow_service import (
    predict_property_investment,
    get_mlflow_client,
    get_or_create_live_experiment,
)


class TestEndToEndPipeline(unittest.TestCase):
    """Automated test suite validating the production inference pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Warm up the model cache once for realistic latency measurement."""
        warmup_payload: Dict[str, Any] = {
            "Property_Type": "Apartment",
            "City": "Bangalore",
            "Locality": "Whitefield",
            "State": "Karnataka",
            "BHK": 3,
            "Size_in_SqFt": 1500.0,
            "Price_in_Lakhs": 125.0,
            "Floor_No": 5,
            "Total_Floors": 14,
            "Year_Built": 2021,
            "Furnished_Status": "Semi-Furnished",
            "Facing": "East",
            "Owner_Type": "First Owner",
            "Availability_Status": "Ready to Move",
            "Amenities": "Gym, Pool, Security",
            "Public_Transport_Accessibility": "High",
            "Parking_Space": "Yes",
            "Security": "Yes",
            "Nearby_Schools": 4,
            "Nearby_Hospitals": 3,
        }
        # Dry-run warm-up without logging to cloud
        _ = predict_property_investment(warmup_payload, log_to_mlflow=False)

    def test_01_unseen_localities_and_fallbacks(self) -> None:
        """Verify pipeline handles completely novel localities and unknown cities gracefully."""
        # Case A: Synthetic unseen locality in a known city
        unseen_loc_payload: Dict[str, Any] = {
            "Property_Type": "Apartment",
            "City": "Bangalore",
            "Locality": "Locality_NonExistent_Alpha_9999",
            "State": "Karnataka",
            "BHK": 2,
            "Size_in_SqFt": 1100.0,
            "Price_in_Lakhs": 95.0,
            "Floor_No": 3,
            "Total_Floors": 10,
            "Year_Built": 2022,
            "Furnished_Status": "Semi-Furnished",
            "Facing": "North",
            "Owner_Type": "Owner",
            "Availability_Status": "Ready to Move",
            "Amenities": "Gym, Security",
            "Public_Transport_Accessibility": "Medium",
            "Parking_Space": "Yes",
            "Security": "Yes",
            "Nearby_Schools": 3,
            "Nearby_Hospitals": 2,
        }
        res_a = predict_property_investment(unseen_loc_payload, log_to_mlflow=False)
        self.assertIn(res_a["prediction"], [0, 1])
        self.assertGreater(res_a["confidence_percent"], 50.0)
        self.assertGreater(res_a["predicted_5y_price_lakhs"], 0.0)
        self.assertFalse(res_a["predicted_5y_price_lakhs"] != res_a["predicted_5y_price_lakhs"])  # Not NaN

        # Case B: Completely unseen city and state
        unseen_city_payload: Dict[str, Any] = {
            "Property_Type": "Independent House",
            "City": "Atlantis_Cyber_City",
            "Locality": "Underwater_Sector_7",
            "State": "Fictional_Territory",
            "BHK": 3,
            "Size_in_SqFt": 1600.0,
            "Price_in_Lakhs": 140.0,
            "Floor_No": 1,
            "Total_Floors": 2,
            "Year_Built": 2023,
            "Furnished_Status": "Furnished",
            "Facing": "East",
            "Owner_Type": "Builder",
            "Availability_Status": "Ready to Move",
            "Amenities": "Pool, Garden, Security",
            "Public_Transport_Accessibility": "High",
            "Parking_Space": "Yes",
            "Security": "Yes",
            "Nearby_Schools": 5,
            "Nearby_Hospitals": 3,
        }
        res_b = predict_property_investment(unseen_city_payload, log_to_mlflow=False)
        self.assertIn(res_b["prediction"], [0, 1])
        self.assertGreater(res_b["predicted_5y_price_lakhs"], 0.0)

    def test_02_extreme_boundary_values(self) -> None:
        """Verify pipeline numerical stability on micro-units and ultra-luxury estates."""
        # Extreme Low: Budget 250 sqft micro-unit
        micro_payload: Dict[str, Any] = {
            "Property_Type": "Apartment",
            "City": "Kolkata",
            "Locality": "Suburban_Outskirts",
            "State": "West Bengal",
            "BHK": 1,
            "Size_in_SqFt": 250.0,
            "Price_in_Lakhs": 12.0,
            "Floor_No": 1,
            "Total_Floors": 2,
            "Year_Built": 1985,
            "Furnished_Status": "Unfurnished",
            "Facing": "West",
            "Owner_Type": "Second Owner",
            "Availability_Status": "Ready to Move",
            "Amenities": "None",
            "Public_Transport_Accessibility": "Low",
            "Parking_Space": "No",
            "Security": "No",
            "Nearby_Schools": 1,
            "Nearby_Hospitals": 0,
        }
        res_micro = predict_property_investment(micro_payload, log_to_mlflow=False)
        self.assertGreater(res_micro["predicted_5y_price_lakhs"], 0.0)
        self.assertIsInstance(res_micro["cagr_percent"], float)

        # Boundary High (Top of Training Distribution): Luxury 4,500 sqft Villa at ₹450.0 Lakhs
        luxury_villa_payload: Dict[str, Any] = {
            "Property_Type": "Villa",
            "City": "Mumbai",
            "Locality": "Worli",
            "State": "Maharashtra",
            "BHK": 5,
            "Size_in_SqFt": 4500.0,
            "Price_in_Lakhs": 450.0,  # Near maximum training distribution (₹500L)
            "Floor_No": 1,
            "Total_Floors": 3,
            "Year_Built": 2024,
            "Furnished_Status": "Furnished",
            "Facing": "North",
            "Owner_Type": "First Owner",
            "Availability_Status": "Ready to Move",
            "Amenities": "Gym, Pool, Clubhouse, Security, Garden, Playground",
            "Public_Transport_Accessibility": "High",
            "Parking_Space": "Yes",
            "Security": "Yes",
            "Nearby_Schools": 8,
            "Nearby_Hospitals": 5,
        }
        res_luxury = predict_property_investment(luxury_villa_payload, log_to_mlflow=False)
        self.assertGreater(res_luxury["predicted_5y_price_lakhs"], 450.0)
        self.assertGreater(res_luxury["appreciation_lakhs"], 0.0)

        # Out-of-Distribution Stress Test: Extreme ₹2,500 Lakhs listing
        extreme_payload = luxury_villa_payload.copy()
        extreme_payload["Price_in_Lakhs"] = 2500.0
        res_extreme = predict_property_investment(extreme_payload, log_to_mlflow=False)
        self.assertGreater(res_extreme["predicted_5y_price_lakhs"], 0.0)
        self.assertIsInstance(res_extreme["confidence_percent"], float)

    def test_03_rapid_query_throughput_and_latency(self) -> None:
        """Measure inference latency across 25 rapid successive requests to verify SLA."""
        cities = ["Bangalore", "Mumbai", "Pune", "Delhi", "Kolkata", "Hyderabad", "Bhubaneswar", "Chennai"]
        latencies: List[float] = []

        for i in range(25):
            city_choice = cities[i % len(cities)]
            test_listing: Dict[str, Any] = {
                "Property_Type": "Apartment",
                "City": city_choice,
                "Locality": f"Locality_Benchmark_{i}",
                "State": "Karnataka" if city_choice == "Bangalore" else "Maharashtra",
                "BHK": (i % 4) + 1,
                "Size_in_SqFt": 800.0 + (i * 50.0),
                "Price_in_Lakhs": 60.0 + (i * 8.0),
                "Floor_No": (i % 15) + 1,
                "Total_Floors": 20,
                "Year_Built": 2018 + (i % 7),
                "Furnished_Status": "Semi-Furnished",
                "Facing": "East",
                "Owner_Type": "First Owner",
                "Availability_Status": "Ready to Move",
                "Amenities": "Gym, Pool, Security",
                "Public_Transport_Accessibility": "High",
                "Parking_Space": "Yes",
                "Security": "Yes",
                "Nearby_Schools": 3,
                "Nearby_Hospitals": 2,
            }
            t0 = time.perf_counter()
            res = predict_property_investment(test_listing, log_to_mlflow=False)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)
            self.assertIn("verdict", res)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        avg = sum(latencies) / len(latencies)

        print(f"\n📊 [Throughput Benchmark] 25 Queries | Avg: {avg:.2f}ms | p50: {p50:.2f}ms | p95: {p95:.2f}ms")
        # Assert sub-100ms p50 latency for production serving
        self.assertLess(p50, 100.0, f"p50 latency {p50:.2f}ms exceeded 100ms threshold")

    def test_04_remote_mlflow_live_audit_logging(self) -> None:
        """Verify real-time remote MLflow audit run is created and queryable on DagsHub."""
        audit_payload: Dict[str, Any] = {
            "Property_Type": "Apartment",
            "City": "Bhubaneswar",
            "Locality": "Patia",
            "State": "Odisha",
            "BHK": 3,
            "Size_in_SqFt": 1400.0,
            "Price_in_Lakhs": 88.0,
            "Floor_No": 5,
            "Total_Floors": 12,
            "Year_Built": 2022,
            "Furnished_Status": "Semi-Furnished",
            "Facing": "East",
            "Owner_Type": "First Owner",
            "Availability_Status": "Ready to Move",
            "Amenities": "Gym, Pool, Security, Garden",
            "Public_Transport_Accessibility": "High",
            "Parking_Space": "Yes",
            "Security": "Yes",
            "Nearby_Schools": 4,
            "Nearby_Hospitals": 2,
        }

        # Execute with live cloud MLflow logging enabled
        result = predict_property_investment(audit_payload, log_to_mlflow=True)
        run_id = result.get("mlflow_run_id")
        self.assertIsNotNone(run_id, "Expected valid mlflow_run_id from live inference call")
        run_id_str = str(run_id)

        # Query DagsHub MLflow Server
        client = get_mlflow_client()
        remote_run = client.get_run(run_id_str)
        self.assertEqual(remote_run.info.run_id, run_id_str)
        self.assertEqual(remote_run.info.status, "FINISHED")

        # Verify logged parameters match request
        params = remote_run.data.params
        self.assertEqual(params.get("City"), "Bhubaneswar")
        self.assertEqual(params.get("Locality"), "Patia")
        self.assertEqual(params.get("BHK"), "3")

        # Verify logged metrics match prediction output
        metrics = remote_run.data.metrics
        self.assertIn("inference_latency_ms", metrics)
        self.assertIn("predicted_5y_price_lakhs", metrics)
        self.assertAlmostEqual(metrics["predicted_5y_price_lakhs"], float(result["predicted_5y_price_lakhs"]), places=1)
        print(f"\n☁️ [DagsHub MLflow Audit] Run {run_id_str[:8]} confirmed on remote tracking server with {len(params)} params and {len(metrics)} metrics.")


if __name__ == "__main__":
    unittest.main()
