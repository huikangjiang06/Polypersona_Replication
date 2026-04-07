"""
Utility functions for MrP analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import json


def create_example_data(
    n_samples: int = 1000,
    output_dir: str = "example_data",
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Create example survey data for testing MrP analysis
    
    Parameters
    ----------
    n_samples : int
        Number of survey respondents to simulate
    output_dir : str
        Directory to save example files
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, str]
        Paths to created files
    """
    np.random.seed(random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Define demographics schema
    demographics_schema = {
        "age_bracket": ["18-29", "30-44", "45-59", "60+"],
        "income_bracket": ["<30k", "30-60k", "60-100k", "100k+"],
        "education": ["high_school", "some_college", "bachelors", "graduate"],
        "region": ["northeast", "south", "midwest", "west"]
    }
    
    # Generate survey data with realistic correlations
    data = []
    for _ in range(n_samples):
        # Sample demographics
        age = np.random.choice(demographics_schema["age_bracket"], p=[0.25, 0.30, 0.25, 0.20])
        income = np.random.choice(demographics_schema["income_bracket"], p=[0.20, 0.35, 0.30, 0.15])
        education = np.random.choice(demographics_schema["education"], p=[0.25, 0.30, 0.25, 0.20])
        region = np.random.choice(demographics_schema["region"], p=[0.20, 0.35, 0.25, 0.20])
        
        # Generate realistic survey responses
        # Binary response (e.g., approve/disapprove)
        approval_prob = 0.5
        if income in ["100k+", "60-100k"]:
            approval_prob += 0.15
        if education in ["graduate", "bachelors"]:
            approval_prob += 0.10
        if region == "west":
            approval_prob += 0.05
        approval = int(np.random.random() < approval_prob)
        
        # Continuous response (e.g., satisfaction score 1-7)
        satisfaction = np.random.normal(4.5, 1.5)
        if age in ["60+", "45-59"]:
            satisfaction += 0.5
        if income in ["100k+", "60-100k"]:
            satisfaction += 0.3
        satisfaction = np.clip(satisfaction, 1, 7)
        satisfaction = int(np.round(satisfaction))
        
        # Count response (e.g., number of times engaged in activity)
        activity_count = np.random.poisson(3)
        if age in ["18-29", "30-44"]:
            activity_count = np.random.poisson(5)
        
        data.append({
            "age_bracket": age,
            "income_bracket": income,
            "education": education,
            "region": region,
            "approval": approval,
            "satisfaction": satisfaction,
            "activity_count": activity_count
        })
    
    df = pd.DataFrame(data)
    
    # Save survey data
    data_path = output_path / "survey_data.csv"
    df.to_csv(data_path, index=False)
    print(f"✓ Created example survey data: {data_path}")
    
    # Save demographics schema
    schema_path = output_path / "demographics_schema.json"
    with open(schema_path, 'w') as f:
        json.dump(demographics_schema, f, indent=2)
    print(f"✓ Created demographics schema: {schema_path}")
    
    # Save response types
    response_types = {
        "approval": "binary",
        "satisfaction": "count",
        "activity_count": "count"
    }
    response_types_path = output_path / "response_types.json"
    with open(response_types_path, 'w') as f:
        json.dump(response_types, f, indent=2)
    print(f"✓ Created response types file: {response_types_path}")
    
    # Create population weights (simulating census data)
    # Generate all demographic combinations
    from itertools import product
    combinations = list(product(*demographics_schema.values()))
    
    poststrat_data = []
    total_population = 10000000  # 10 million population
    
    for combo in combinations:
        # Assign realistic population counts
        base_count = total_population / len(combinations)
        
        # Adjust based on known demographic distributions
        age, income, edu, reg = combo
        adjustment = 1.0
        
        # More young people
        if age == "18-29":
            adjustment *= 1.2
        # Fewer very old
        elif age == "60+":
            adjustment *= 0.8
        
        # Income distribution
        if income == "<30k":
            adjustment *= 1.3
        elif income == "100k+":
            adjustment *= 0.7
        
        pop_count = int(base_count * adjustment)
        
        poststrat_data.append({
            "age_bracket": age,
            "income_bracket": income,
            "education": edu,
            "region": reg,
            "population_count": pop_count
        })
    
    poststrat_df = pd.DataFrame(poststrat_data)
    
    # Normalize to exactly match total population
    poststrat_df['population_count'] = (
        poststrat_df['population_count'] / poststrat_df['population_count'].sum() * total_population
    ).astype(int)
    
    weights_path = output_path / "population_weights.csv"
    poststrat_df.to_csv(weights_path, index=False)
    print(f"✓ Created population weights: {weights_path}")
    
    print(f"\nExample files created in {output_path}/")
    print("\nTo run MrP analysis on example data, use:")
    print(f"python mrp_analysis.py {data_path} {schema_path} {response_types_path} --population-weights {weights_path}")
    
    return {
        "data": str(data_path),
        "schema": str(schema_path),
        "response_types": str(response_types_path),
        "weights": str(weights_path)
    }


def plot_subgroup_estimates(
    estimates_path: str,
    demographic_var: str,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Create visualization of subgroup estimates for a specific demographic variable
    
    Parameters
    ----------
    estimates_path : str
        Path to CSV file with subgroup estimates
    demographic_var : str
        Demographic variable to plot
    output_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    df = pd.read_csv(estimates_path)
    
    # Aggregate by demographic variable
    grouped = df.groupby(demographic_var).agg({
        'estimate_mean': 'mean',
        'ci_lower': 'mean',
        'ci_upper': 'mean',
        'population_count': 'sum'
    }).reset_index()
    
    # Sort by estimate
    grouped = grouped.sort_values('estimate_mean')
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot estimates with confidence intervals
    ax1.errorbar(
        grouped['estimate_mean'],
        range(len(grouped)),
        xerr=[
            grouped['estimate_mean'] - grouped['ci_lower'],
            grouped['ci_upper'] - grouped['estimate_mean']
        ],
        fmt='o',
        markersize=8,
        capsize=5
    )
    ax1.set_yticks(range(len(grouped)))
    ax1.set_yticklabels(grouped[demographic_var])
    ax1.set_xlabel('Estimate (95% CI)')
    ax1.set_title(f'Estimates by {demographic_var}')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot population distribution
    ax2.barh(range(len(grouped)), grouped['population_count'])
    ax2.set_yticks(range(len(grouped)))
    ax2.set_yticklabels(grouped[demographic_var])
    ax2.set_xlabel('Population Count')
    ax2.set_title(f'Population Distribution by {demographic_var}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {output_path}")
    else:
        plt.show()


def compare_naive_vs_mrp(
    data_path: str,
    mrp_results_path: str,
    response_var: str,
    output_path: Optional[str] = None
):
    """
    Compare naive sample estimates with MrP-adjusted estimates
    
    Parameters
    ----------
    data_path : str
        Path to original survey data
    mrp_results_path : str
        Path to MrP aggregate estimates
    response_var : str
        Response variable to compare
    output_path : str, optional
        Path to save comparison plot
    """
    # Load data
    df = pd.read_csv(data_path)
    mrp_df = pd.read_csv(mrp_results_path)
    
    # Calculate naive estimate (simple mean)
    naive_estimate = df[response_var].mean()
    naive_std = df[response_var].std() / np.sqrt(len(df))
    
    # Get MrP estimate
    mrp_row = mrp_df[mrp_df['response_variable'] == response_var].iloc[0]
    mrp_estimate = mrp_row['estimate_mean']
    mrp_ci_lower = mrp_row['ci_lower']
    mrp_ci_upper = mrp_row['ci_upper']
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = ['Naive\n(Sample Mean)', 'MrP\n(Poststratified)']
    estimates = [naive_estimate, mrp_estimate]
    errors_lower = [naive_std * 1.96, mrp_estimate - mrp_ci_lower]
    errors_upper = [naive_std * 1.96, mrp_ci_upper - mrp_estimate]
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, estimates, color=['lightcoral', 'skyblue'], alpha=0.7)
    ax.errorbar(x_pos, estimates, yerr=[errors_lower, errors_upper],
                fmt='none', color='black', capsize=10, linewidth=2)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel(f'{response_var} Estimate')
    ax.set_title(f'Comparison: Naive vs MrP Estimates for {response_var}')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (est, method) in enumerate(zip(estimates, methods)):
        ax.text(i, est, f'{est:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {output_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Comparison for {response_var}")
    print(f"{'='*50}")
    print(f"Naive estimate:     {naive_estimate:.4f} (SE: {naive_std:.4f})")
    print(f"MrP estimate:       {mrp_estimate:.4f} (95% CI: [{mrp_ci_lower:.4f}, {mrp_ci_upper:.4f}])")
    print(f"Difference:         {abs(mrp_estimate - naive_estimate):.4f}")
    print(f"Relative change:    {100*(mrp_estimate - naive_estimate)/naive_estimate:.2f}%")


def create_summary_report(
    results_dir: str,
    output_path: Optional[str] = None
):
    """
    Create a summary report of MrP analysis results
    
    Parameters
    ----------
    results_dir : str
        Directory containing MrP results
    output_path : str, optional
        Path to save report (markdown format)
    """
    results_path = Path(results_dir)
    
    # Load aggregate results
    aggregate_path = results_path / "aggregate_estimates.json"
    with open(aggregate_path, 'r') as f:
        aggregate_results = json.load(f)
    
    # Create report
    report = ["# MrP Analysis Summary Report\n"]
    report.append(f"**Results Directory:** `{results_dir}`\n")
    report.append(f"**Number of Response Variables:** {len(aggregate_results)}\n")
    
    report.append("\n## Population-Level Estimates\n")
    report.append("| Response Variable | Type | Estimate (Mean) | 95% CI | Std Dev |")
    report.append("|------------------|------|-----------------|--------|---------|")
    
    for result in aggregate_results:
        var = result['response_variable']
        rtype = result['response_type']
        mean = result['estimate_mean']
        ci_low = result['ci_lower']
        ci_high = result['ci_upper']
        std = result['estimate_std']
        
        report.append(
            f"| {var} | {rtype} | {mean:.4f} | "
            f"[{ci_low:.4f}, {ci_high:.4f}] | {std:.4f} |"
        )
    
    report.append("\n## Files Generated\n")
    report.append("- `aggregate_estimates.csv` - Population-level estimates for all response variables")
    report.append("- `aggregate_estimates.json` - Same as above in JSON format")
    report.append("- `poststratification_table.csv` - Full table of demographic strata with population weights")
    
    for result in aggregate_results:
        var = result['response_variable']
        report.append(f"- `subgroup_estimates_{var}.csv` - Subgroup estimates for {var}")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✓ Saved summary report to {output_path}")
    else:
        print(report_text)
    
    return report_text


if __name__ == "__main__":
    # Generate example data when run directly
    print("Generating example data for MrP analysis...\n")
    paths = create_example_data(n_samples=1000, output_dir="example_data")
    print("\n✓ Example data generation complete!")
