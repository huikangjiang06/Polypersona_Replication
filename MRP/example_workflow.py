"""
Complete MrP Analysis Example
==============================

This script demonstrates the full workflow of Multilevel Regression 
and Poststratification (MrP) analysis from data generation to results.
"""

import sys
from pathlib import Path

# Add MRP directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mrp_utils import (
    create_example_data,
    plot_subgroup_estimates,
    compare_naive_vs_mrp,
    create_summary_report
)
from mrp_analysis import MrPAnalyzer
import pandas as pd


def main():
    print("="*70)
    print("MULTILEVEL REGRESSION AND POSTSTRATIFICATION (MrP)")
    print("Complete Example Workflow")
    print("="*70)
    
    # Step 1: Generate example data
    print("\n" + "="*70)
    print("STEP 1: Generating Example Survey Data")
    print("="*70)
    
    example_paths = create_example_data(
        n_samples=1000,
        output_dir="example_data",
        random_seed=42
    )
    
    # Step 2: Initialize MrP analyzer
    print("\n" + "="*70)
    print("STEP 2: Initializing MrP Analyzer")
    print("="*70)
    
    analyzer = MrPAnalyzer(
        data_path=example_paths["data"],
        demographics_schema_path=example_paths["schema"],
        output_dir="mrp_results"
    )
    
    # Step 3: Load population weights
    print("\n" + "="*70)
    print("STEP 3: Loading Population Weights")
    print("="*70)
    
    population_weights = pd.read_csv(example_paths["weights"])
    print(f"Loaded population weights for {len(population_weights)} demographic strata")
    print(f"Total population: {population_weights['population_count'].sum():,.0f}")
    
    # Step 4: Define response types
    print("\n" + "="*70)
    print("STEP 4: Defining Response Variable Types")
    print("="*70)
    
    response_types = {
        "approval": "binary",
        "satisfaction": "count",
        "activity_count": "count"
    }
    
    print("Response variables:")
    for var, rtype in response_types.items():
        print(f"  - {var}: {rtype}")
    
    # Step 5: Run MrP analysis
    print("\n" + "="*70)
    print("STEP 5: Running MrP Analysis")
    print("="*70)
    print("This may take a few minutes...")
    
    analyzer.run_full_analysis(
        response_types=response_types,
        population_weights=population_weights,
        draws=2000,
        target_accept=0.95,
        random_seed=42
    )
    
    # Step 6: Generate visualizations
    print("\n" + "="*70)
    print("STEP 6: Creating Visualizations")
    print("="*70)
    
    # Plot subgroup estimates for each response variable
    for response_var in response_types.keys():
        if Path(f"mrp_results/subgroup_estimates_{response_var}.csv").exists():
            print(f"\nCreating plots for {response_var}...")
            
            # Plot by age bracket
            plot_subgroup_estimates(
                estimates_path=f"mrp_results/subgroup_estimates_{response_var}.csv",
                demographic_var="age_bracket",
                output_path=f"mrp_results/plot_{response_var}_by_age.png"
            )
            
            # Plot by region
            plot_subgroup_estimates(
                estimates_path=f"mrp_results/subgroup_estimates_{response_var}.csv",
                demographic_var="region",
                output_path=f"mrp_results/plot_{response_var}_by_region.png"
            )
    
    # Step 7: Compare naive vs MrP estimates
    print("\n" + "="*70)
    print("STEP 7: Comparing Naive vs MrP Estimates")
    print("="*70)
    
    for response_var in response_types.keys():
        print(f"\nComparing {response_var}...")
        compare_naive_vs_mrp(
            data_path=example_paths["data"],
            mrp_results_path="mrp_results/aggregate_estimates.csv",
            response_var=response_var,
            output_path=f"mrp_results/comparison_{response_var}.png"
        )
    
    # Step 8: Generate summary report
    print("\n" + "="*70)
    print("STEP 8: Generating Summary Report")
    print("="*70)
    
    create_summary_report(
        results_dir="mrp_results",
        output_path="mrp_results/SUMMARY_REPORT.md"
    )
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nResults saved in: mrp_results/")
    print("\nKey files:")
    print("  - aggregate_estimates.csv - Population-level estimates")
    print("  - subgroup_estimates_*.csv - Subgroup-level estimates")
    print("  - comparison_*.png - Naive vs MrP comparisons")
    print("  - plot_*_by_*.png - Subgroup visualizations")
    print("  - SUMMARY_REPORT.md - Analysis summary")
    print("\nTo view aggregate estimates:")
    print("  cat mrp_results/aggregate_estimates.csv")
    print("\nTo view summary report:")
    print("  cat mrp_results/SUMMARY_REPORT.md")
    print("="*70)


if __name__ == "__main__":
    main()
