"""
Compare script to create CSV comparing reference and prediction texts.
Loads reference from generated_data and predictions from evaluation_results.
"""

import json
import argparse
import pandas as pd
from pathlib import Path


def load_json_file(path):
    """Load JSONL or JSON file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Try as full JSON
                f.seek(0)
                records = json.load(f)
                break
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference and prediction texts in CSV format"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to compare (default: test)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./generated_data",
        help="Directory containing reference data"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="./evaluation_results",
        help="Directory containing prediction data"
    )
    
    args = parser.parse_args()
    
    # Load reference data
    reference_path = Path(args.data_dir) / f"{args.split}.json"
    print(f"Loading references from: {reference_path}")
    reference_data = load_json_file(reference_path)
    print(f"Loaded {len(reference_data)} reference examples")
    
    # Load prediction data
    prediction_path = Path(args.eval_dir) / f"{args.split}_predictions.jsonl"
    print(f"Loading predictions from: {prediction_path}")
    prediction_data = load_json_file(prediction_path)
    print(f"Loaded {len(prediction_data)} prediction examples")
    
    # Create dictionary for quick lookup by ID
    reference_dict = {item['id']: item.get('reference', '') for item in reference_data}
    prediction_dict = {item['id']: item.get('prediction', '') for item in prediction_data}
    
    # Match references and predictions by ID
    rows = []
    for item_id in reference_dict.keys():
        reference = reference_dict.get(item_id, '')
        prediction = prediction_dict.get(item_id, '')
        
        rows.append({
            'id': item_id,
            'reference': reference,
            'prediction': prediction
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = Path(args.eval_dir) / f"compare_{args.split}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nSaved comparison to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
