"""
predict.py
----------
Standalone command-line interface for generating late delivery predictions.

Usage:
    # Interactive prompt mode:
    python predict.py --interactive

    # Batch CSV prediction mode:
    python predict.py --batch path/to/orders.csv --output path/to/predictions.csv

    # Single order prediction with demo payload:
    python predict.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_logger
from src.prediction import Predictor

logger = get_logger("predict_cli")

DEMO_ORDER = {
    "purchase_month": 11,
    "purchase_day_of_week": 4,
    "purchase_hour": 15,
    "estimated_delivery_days": 18,
    "product_weight_g": 1200.0,
    "product_volume_cm3": 8500.0,
    "freight_value": 34.50,
    "price": 189.90,
    "freight_to_price_ratio": 0.1817,
    "n_items": 1,
    "n_sellers": 1,
    "payment_installments": 4,
    "total_payment_value": 224.40,
    "geo_distance_km": 620.0,
    "product_category_name_english": "computers_accessories",
    "seller_state": "SP",
    "customer_state": "RJ",
    "payment_type": "credit_card",
}


def run_demo(predictor: Predictor):
    print("\n" + "=" * 60)
    print("DEMO ORDER PREDICTION")
    print("=" * 60)
    print("Input Order Payload:")
    print(json.dumps(DEMO_ORDER, indent=2))
    print("\nRunning inference...")
    res = predictor.predict_single(DEMO_ORDER)
    print("\nResult:")
    print(f"  Class:       {res['predicted_class']} ({'LATE' if res['predicted_class'] == 1 else 'ON TIME'})")
    print(f"  Probability: {res['late_probability']:.2%}")
    print(f"  Risk Level:  {res['risk_level']}")
    print(f"  Message:     {res['message']}")
    print("=" * 60 + "\n")


def run_batch(predictor: Predictor, input_path: str, output_path: str | None):
    print(f"\nLoading batch CSV: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Found {len(df)} records. Predicting...")
    results_df = predictor.predict_batch(df)

    out_file = output_path or "batch_predictions_output.csv"
    results_df.to_csv(out_file, index=False)
    print(f"Batch prediction complete! Results saved to: {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="E-commerce Late Delivery Predictor CLI")
    parser.add_argument("--demo", action="store_true", help="Run prediction on a sample demo order")
    parser.add_argument("--batch", type=str, help="Path to input CSV file for batch predictions")
    parser.add_argument("--output", type=str, help="Output path for batch predictions CSV")
    args = parser.parse_args()

    predictor = Predictor()

    if args.demo:
        run_demo(predictor)
    elif args.batch:
        run_batch(predictor, args.batch, args.output)
    else:
        # Default to demo if no args provided
        print("No mode specified. Running demo order prediction...")
        run_demo(predictor)


if __name__ == "__main__":
    main()

