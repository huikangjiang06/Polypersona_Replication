#!/usr/bin/env python3
"""
Subgroup Analysis Script for PolyPersona Evaluation

Creates demographic subgroup analysis similar to Table 2 from:
"Assessing the Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents"
(arXiv:2602.18462)

This script analyzes model performance across demographic subgroups by:
1. Loading synthetic persona data and evaluation predictions
2. Grouping personas by demographic attributes (age, gender, education, etc.)
3. Computing aggregate metrics (BLEU, ROUGE, BERTScore) for each subgroup
4. Identifying performance disparities across subgroups

Usage:
    # Basic usage with defaults
    python subgroup_report.py
    
    # Specify custom directories
    python subgroup_report.py --data-dir outputs/experiment_1_synthetic_data \
                              --pred-dir outputs/experiment_1_results \
                              --output-dir outputs/experiment_1_subgroup_analysis
    
    # Analyze only test split
    python subgroup_report.py --splits test

Output:
    - {split}_subgroup_metrics.csv: CSV file with metrics for each demographic subgroup
      Columns: attribute, value, n_personas, bleu_mean, bleu_std, rouge1_mean, 
               rouge1_std, rouge2_mean, rouge2_std, rougeL_mean, rougeL_std,
               bertscore_f1_mean, bertscore_f1_std
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys


def load_jsonl(path):
    """Load JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json(path):
    """Load JSON or JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        # Try JSON array first
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
        except json.JSONDecodeError:
            # Fall back to JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records


def categorize_age(age):
    """Categorize age into brackets."""
    if age < 20:
        return "<20"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60+"


def get_demographic_slices(persona):
    """
    Extract demographic slices from persona dict.
    
    Returns dict of {attribute_name: attribute_value}
    """
    slices = {}
    
    # Age bracket
    if 'age' in persona:
        slices['age_group'] = categorize_age(persona['age'])
    
    # Gender
    if 'gender' in persona:
        slices['gender'] = persona['gender']
    
    # Education level
    if 'education' in persona:
        slices['education'] = persona['education']
    
    # Region
    if 'region' in persona:
        slices['region'] = persona['region']
    
    # Occupation
    if 'occupation' in persona:
        slices['occupation'] = persona['occupation']
    
    # Income bracket
    if 'income_bracket' in persona:
        slices['income_bracket'] = persona['income_bracket']
    
    return slices


def compute_subgroup_metrics(data_with_metrics, demographic_key):
    """
    Compute aggregated metrics for each value of a demographic attribute.
    
    Args:
        data_with_metrics: List of dicts with 'persona' and metrics (bleu, rouge1, etc.)
        demographic_key: Key to group by (e.g., 'age_group', 'gender')
    
    Returns:
        DataFrame with subgroup statistics
    """
    # Group by demographic value
    groups = defaultdict(list)
    
    for item in data_with_metrics:
        persona = item.get('persona', {})
        slices = get_demographic_slices(persona)
        
        if demographic_key not in slices:
            continue
            
        demographic_value = slices[demographic_key]
        groups[demographic_value].append(item)
    
    # Compute statistics for each group
    results = []
    metric_names = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    
    for value, items in sorted(groups.items()):
        row = {
            'attribute': demographic_key,
            'value': value,
            'n_personas': len(items)
        }
        
        # Compute mean for each metric
        for metric in metric_names:
            values = [item[metric] for item in items if metric in item]
            if values:
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)


def plot_ascii_bar(values, labels, title, metric_name, width=50, show_values=True):
    """
    Create an ASCII bar chart.
    
    Args:
        values: List of values to plot
        labels: List of labels for each bar
        title: Chart title
        metric_name: Name of the metric being plotted
        width: Width of the chart in characters
        show_values: Whether to show numeric values
    """
    if not values or all(np.isnan(values)):
        return
    
    # Filter out NaN values
    valid_data = [(v, l) for v, l in zip(values, labels) if not np.isnan(v)]
    if not valid_data:
        return
    
    values, labels = zip(*valid_data)
    values = list(values)
    labels = list(labels)
    
    max_val = max(values)
    min_val = min(values)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Calculate overall mean
    mean_val = np.mean(values)
    
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"Metric: {metric_name}")
    print(f"Mean: {mean_val:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f}")
    print(f"{'=' * 70}")
    
    # Find max label length for alignment
    max_label_len = max(len(str(l)) for l in labels)
    
    for label, value in zip(labels, values):
        # Calculate bar length
        if val_range > 0:
            bar_len = int((value - min_val) / val_range * width)
        else:
            bar_len = width // 2
        
        # Determine bar character based on performance relative to mean
        if value >= mean_val:
            bar_char = '█'
            bar_color = ''
        else:
            bar_char = '░'
            bar_color = ''
        
        bar = bar_char * bar_len
        
        # Format the line
        label_str = f"{str(label):<{max_label_len}}"
        if show_values:
            print(f"  {label_str} │ {bar} {value:.4f}")
        else:
            print(f"  {label_str} │ {bar}")
    
    print()


def plot_ascii_deviation(values, labels, title, metric_name, width=40):
    """
    Create an ASCII deviation chart showing distance from mean.
    
    Args:
        values: List of values to plot
        labels: List of labels for each bar
        title: Chart title
        metric_name: Name of the metric being plotted
        width: Width of the chart in characters (for each side)
    """
    if not values or all(np.isnan(values)):
        return
    
    # Filter out NaN values
    valid_data = [(v, l) for v, l in zip(values, labels) if not np.isnan(v)]
    if not valid_data:
        return
    
    values, labels = zip(*valid_data)
    values = list(values)
    labels = list(labels)
    
    mean_val = np.mean(values)
    deviations = [v - mean_val for v in values]
    max_abs_dev = max(abs(d) for d in deviations) if deviations else 1
    
    print(f"\n{'=' * 70}")
    print(f"{title} - Deviation from Mean")
    print(f"Metric: {metric_name} (Mean: {mean_val:.4f})")
    print(f"{'=' * 70}")
    
    # Find max label length for alignment
    max_label_len = max(len(str(l)) for l in labels)
    
    # Sort by deviation for better visualization
    sorted_data = sorted(zip(deviations, labels, values), reverse=True)
    
    for dev, label, value in sorted_data:
        if max_abs_dev > 0:
            bar_len = int(abs(dev) / max_abs_dev * width)
        else:
            bar_len = 0
        
        label_str = f"{str(label):<{max_label_len}}"
        
        if dev >= 0:
            # Positive deviation (better than mean)
            bar = '█' * bar_len
            print(f"  {label_str} │{' ' * width}│{bar} +{dev:.4f}")
        else:
            # Negative deviation (worse than mean)
            bar = '░' * bar_len
            spaces = ' ' * (width - bar_len)
            print(f"  {label_str} │{spaces}{bar}│ {dev:.4f}")
    
    # Print the zero line indicator
    print(f"  {' ' * max_label_len} │{' ' * width}│")
    print(f"  {' ' * max_label_len}   {'←' + ' worse':<{width}}  better →")
    print()


def visualize_subgroup_metrics(df, split_name):
    """
    Create visualizations for subgroup metrics.
    
    Args:
        df: DataFrame with subgroup metrics
        split_name: Name of the split (test/val)
    """
    print(f"\n\n{'#' * 70}")
    print(f"# SUBGROUP PERFORMANCE VISUALIZATION - {split_name.upper()} SPLIT")
    print(f"{'#' * 70}\n")
    
    # Get unique attributes
    attributes = df['attribute'].unique()
    
    for attr in attributes:
        attr_df = df[df['attribute'] == attr].copy()
        
        if attr_df.empty:
            continue
        
        # Sort by value for consistent display
        attr_df = attr_df.sort_values('value')
        
        labels = attr_df['value'].tolist()
        
        # 1. BERTScore F1 bar chart
        bertscore_values = attr_df['bertscore_f1_mean'].tolist()
        plot_ascii_bar(
            bertscore_values,
            labels,
            f"{attr.upper().replace('_', ' ')} - BERTScore F1",
            "BERTScore F1"
        )
        
        # 2. BERTScore F1 deviation chart
        plot_ascii_deviation(
            bertscore_values,
            labels,
            f"{attr.upper().replace('_', ' ')}",
            "BERTScore F1"
        )
        
        # 3. ROUGE-L bar chart (compact)
        rougeL_values = attr_df['rougeL_mean'].tolist()
        if not all(np.isnan(rougeL_values)):
            plot_ascii_bar(
                rougeL_values,
                labels,
                f"{attr.upper().replace('_', ' ')} - ROUGE-L",
                "ROUGE-L",
                width=40,
                show_values=True
            )
    
    # Summary comparison across all attributes
    print(f"\n{'#' * 70}")
    print(f"# CROSS-ATTRIBUTE PERFORMANCE SUMMARY")
    print(f"{'#' * 70}\n")
    
    # Find best and worst subgroups overall
    df_sorted = df.sort_values('bertscore_f1_mean', ascending=False)
    
    print("Top 5 Performing Subgroups (BERTScore F1):")
    print("-" * 70)
    for idx, row in df_sorted.head(5).iterrows():
        attr_display = row['attribute'].replace('_', ' ').title()
        print(f"  {attr_display:20s} | {row['value']:20s} | "
              f"n={int(row['n_personas']):4d} | F1={row['bertscore_f1_mean']:.4f}")
    
    print("\nBottom 5 Performing Subgroups (BERTScore F1):")
    print("-" * 70)
    for idx, row in df_sorted.tail(5).iterrows():
        attr_display = row['attribute'].replace('_', ' ').title()
        print(f"  {attr_display:20s} | {row['value']:20s} | "
              f"n={int(row['n_personas']):4d} | F1={row['bertscore_f1_mean']:.4f}")
    
    print()


def merge_data_and_predictions(data_dir, pred_file, split_name):
    """
    Merge synthetic data with predictions.
    
    Args:
        data_dir: Path to directory containing synthetic data files
        pred_file: Path to predictions JSONL file
        split_name: 'test' or 'val'
    
    Returns:
        List of merged records
    """
    # Load synthetic data
    data_path = Path(data_dir) / f"{split_name}.json"
    if not data_path.exists():
        print(f"Warning: {data_path} not found, skipping {split_name}")
        return []
    
    synthetic_data = load_json(data_path)
    
    # Create ID to persona mapping
    id_to_record = {item['id']: item for item in synthetic_data}
    
    # Load predictions
    if not Path(pred_file).exists():
        print(f"Warning: {pred_file} not found, skipping {split_name}")
        return []
    
    predictions = load_jsonl(pred_file)
    
    # Merge by ID
    merged = []
    for pred in predictions:
        item_id = pred.get('id')
        if item_id in id_to_record:
            record = id_to_record[item_id].copy()
            # Add prediction metrics
            record.update({
                'prediction': pred.get('prediction'),
                'bleu': pred.get('bleu'),
                'rouge1': pred.get('rouge1'),
                'rouge2': pred.get('rouge2'),
                'rougeL': pred.get('rougeL'),
                'bertscore_f1': pred.get('bertscore_f1')
            })
            merged.append(record)
    
    print(f"Merged {len(merged)} records for {split_name}")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Generate subgroup analysis report for PolyPersona evaluation"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='outputs/experiment_1_synthetic_data',
        help='Directory containing synthetic data files (train.json, val.json, test.json)'
    )
    parser.add_argument(
        '--pred-dir',
        type=str,
        default='outputs/experiment_1_results',
        help='Directory containing prediction files (test_predictions.jsonl, val_predictions.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output reports (default: same as pred-dir)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['test', 'val'],
        help='Splits to analyze (default: test val)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.pred_dir
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        
        pred_file = Path(args.pred_dir) / f"{split}_predictions.jsonl"
        merged_data = merge_data_and_predictions(args.data_dir, pred_file, split)
        
        if not merged_data:
            print(f"No data for {split}, skipping")
            continue
        
        # Demographic attributes to analyze
        demographic_attrs = [
            'age_group',
            'gender',
            'education',
            'region',
            'occupation',
            'income_bracket'
        ]
        
        # Compute subgroup metrics for each demographic attribute
        all_results = []
        
        for attr in demographic_attrs:
            print(f"\nAnalyzing {attr}...")
            df = compute_subgroup_metrics(merged_data, attr)
            if not df.empty:
                all_results.append(df)
                print(f"  Found {len(df)} subgroups")
        
        if all_results:
            # Combine all results
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Save to CSV
            output_file = output_dir / f"{split}_subgroup_metrics.csv"
            combined_df.to_csv(output_file, index=False, float_format='%.4f')
            print(f"\nSaved subgroup analysis to: {output_file}")
            
            # Visualize the results
            visualize_subgroup_metrics(combined_df, split)
            
            # Print summary statistics
            print(f"\n{'='*60}")
            print(f"Summary for {split} split")
            print(f"{'='*60}")
            
            for attr in demographic_attrs:
                attr_df = combined_df[combined_df['attribute'] == attr]
                if not attr_df.empty:
                    print(f"\n{attr.upper()}:")
                    print(f"  Subgroups: {len(attr_df)}")
                    print(f"  Total personas: {attr_df['n_personas'].sum()}")
                    
                    # Show top and bottom performers by BERTScore
                    if 'bertscore_f1_mean' in attr_df.columns:
                        sorted_df = attr_df.sort_values('bertscore_f1_mean', ascending=False)
                        
                        print(f"\n  Top 3 subgroups by BERTScore F1:")
                        for _, row in sorted_df.head(3).iterrows():
                            print(f"    {row['value']:20s} (n={int(row['n_personas']):4d}): {row['bertscore_f1_mean']:.4f}")
                        
                        print(f"\n  Bottom 3 subgroups by BERTScore F1:")
                        for _, row in sorted_df.tail(3).iterrows():
                            print(f"    {row['value']:20s} (n={int(row['n_personas']):4d}): {row['bertscore_f1_mean']:.4f}")
    
    print(f"\n{'='*60}")
    print("Subgroup analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
