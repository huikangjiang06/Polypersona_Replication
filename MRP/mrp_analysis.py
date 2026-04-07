"""
Multilevel Regression and Poststratification (MrP) Analysis

This script implements the classical MrP approach for survey data analysis,
following the methodology from:
- Gelman & Little (1997) - Poststratification into many categories using hierarchical logistic regression
- Park, Gelman & Bafumi (2004) - Bayesian multilevel estimation with poststratification
"""

import argparse
import json
import pandas as pd
import numpy as np
import bambi as bmb
import arviz as az
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MrPAnalyzer:
    """
    Multilevel Regression and Poststratification Analyzer
    
    Implements the MrP method for adjusting non-representative survey samples
    using demographic information and population counts.
    """
    
    def __init__(
        self, 
        data_path: str, 
        demographics_schema_path: str,
        output_dir: str = "mrp_results"
    ):
        """
        Initialize MrP analyzer
        
        Parameters
        ----------
        data_path : str
            Path to CSV file containing individual-level survey data
        demographics_schema_path : str
            Path to JSON file specifying valid demographic categories
        output_dir : str
            Directory to save output files
        """
        self.data_path = data_path
        self.demographics_schema_path = demographics_schema_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load and validate data
        print("Loading data...")
        self.data = pd.read_csv(data_path)
        with open(demographics_schema_path, 'r') as f:
            self.demographics_schema = json.load(f)
        
        print("Validating demographics...")
        self.validate_demographics()
        
        # Identify demographic and response columns
        self.demographic_cols = list(self.demographics_schema.keys())
        self.response_cols = [col for col in self.data.columns 
                             if col not in self.demographic_cols]
        
        print(f"Found {len(self.demographic_cols)} demographic variables: {self.demographic_cols}")
        print(f"Found {len(self.response_cols)} response variables: {self.response_cols}")
        
        self.models = {}
        self.results = {}
        
    def validate_demographics(self):
        """
        Validate that all demographic values in data match the schema
        """
        for demographic, valid_values in self.demographics_schema.items():
            if demographic not in self.data.columns:
                raise ValueError(f"Demographic '{demographic}' from schema not found in data columns")
            
            data_values = set(self.data[demographic].dropna().unique())
            schema_values = set(valid_values)
            
            invalid_values = data_values - schema_values
            if invalid_values:
                raise ValueError(
                    f"Invalid values found in '{demographic}': {invalid_values}\n"
                    f"Valid values according to schema: {schema_values}"
                )
        
        print("✓ All demographics validated successfully")
    
    def prepare_poststratification_table(
        self, 
        population_weights: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create poststratification table with all demographic combinations
        
        Parameters
        ----------
        population_weights : pd.DataFrame, optional
            DataFrame with demographic combinations and their population counts.
            If None, uses equal weights (assuming representative sample).
            
        Returns
        -------
        pd.DataFrame
            Poststratification table with demographic combinations and weights
        """
        # Create all possible combinations of demographics
        demographic_combinations = []
        for col in self.demographic_cols:
            demographic_combinations.append(self.demographics_schema[col])
        
        # Generate Cartesian product
        from itertools import product
        all_combinations = list(product(*demographic_combinations))
        
        poststrat_df = pd.DataFrame(
            all_combinations, 
            columns=self.demographic_cols
        )
        
        if population_weights is not None:
            # Merge with provided population weights
            poststrat_df = poststrat_df.merge(
                population_weights, 
                on=self.demographic_cols, 
                how='left'
            )
            
            # Fill missing weights with 0 (strata not in population)
            if 'population_count' not in poststrat_df.columns:
                raise ValueError("population_weights must have 'population_count' column")
            poststrat_df['population_count'] = poststrat_df['population_count'].fillna(0)
        else:
            # Use observed sample proportions as proxy for population
            sample_counts = self.data.groupby(self.demographic_cols).size().reset_index(name='population_count')
            poststrat_df = poststrat_df.merge(
                sample_counts,
                on=self.demographic_cols,
                how='left'
            )
            poststrat_df['population_count'] = poststrat_df['population_count'].fillna(0)
        
        # Calculate proportions
        total_pop = poststrat_df['population_count'].sum()
        if total_pop > 0:
            poststrat_df['proportion'] = poststrat_df['population_count'] / total_pop
        else:
            raise ValueError("Total population count is 0. Cannot calculate proportions.")
        
        print(f"Created poststratification table with {len(poststrat_df)} strata")
        print(f"Total population: {total_pop:,.0f}")
        
        return poststrat_df
    
    def fit_multilevel_model(
        self, 
        response_col: str,
        response_type: str = "continuous",
        draws: int = 2000,
        target_accept: float = 0.95,
        random_seed: int = 42
    ):
        """
        Fit multilevel regression model with hierarchical priors
        
        Parameters
        ----------
        response_col : str
            Name of response variable to model
        response_type : str
            Type of response: "continuous", "binary", "count", or "ordinal"
        draws : int
            Number of MCMC samples to draw
        target_accept : float
            Target acceptance rate for MCMC sampler
        random_seed : int
            Random seed for reproducibility
        """
        print(f"\n{'='*60}")
        print(f"Fitting MrP model for: {response_col}")
        print(f"{'='*60}")
        
        # Prepare data - remove missing values
        model_data = self.data[[response_col] + self.demographic_cols].dropna()
        
        # Check if we have enough data
        if len(model_data) < 10:
            print(f"  ⚠ Warning: Only {len(model_data)} observations. Skipping this variable.")
            print(f"  Minimum 10 observations recommended for stable estimates.")
            return None, None
        
        print(f"Using {len(model_data)} complete observations")
        
        # Convert categorical variables to category dtype for Bambi
        for col in self.demographic_cols:
            model_data[col] = pd.Categorical(
                model_data[col], 
                categories=self.demographics_schema[col]
            )
        
        # For binary responses, ensure values are exactly 0 or 1
        if response_type == "binary":
            model_data[response_col] = model_data[response_col].round().astype(int)
            if not set(model_data[response_col].unique()).issubset({0, 1}):
                print(f"  ⚠ Warning: Binary variable contains values other than 0/1. Skipping.")
                return None, None
        
        # Build formula with random intercepts for each demographic
        # Following MrP standard: y ~ 1 + (1|demographic1) + (1|demographic2) + ...
        random_effects = " + ".join([f"(1|{col})" for col in self.demographic_cols])
        formula = f"{response_col} ~ 1 + {random_effects}"
        
        print(f"Model formula: {formula}")
        
        # Determine family based on response type
        if response_type == "binary":
            family = "bernoulli"
        elif response_type == "count":
            family = "poisson"
        elif response_type == "ordinal":
            # For ordinal responses (Likert scales), treat as count
            # This preserves ordering while being robust to sparse data
            # Alternative: use cumulative family with proper data formatting
            family = "poisson"
            print(f"  Note: Treating ordinal as count (preserves ordering)")
        else:  # continuous
            family = "gaussian"
        
        print(f"Using {family} family for {response_type} response")
        
        # Fit model using Bambi
        print("Fitting hierarchical model with Bambi...")
        model = bmb.Model(formula, model_data, family=family)
        
        # Display model structure
        print("\nModel structure:")
        print(model)
        
        # Fit model
        result = model.fit(
            draws=draws,
            target_accept=target_accept,
            random_seed=random_seed,
            chains=4
        )
        
        # Store model and results
        self.models[response_col] = model
        self.results[response_col] = result
        
        # Print diagnostics
        print("\nModel diagnostics:")
        print(az.summary(result, var_names=['Intercept']))
        
        # Check convergence (R-hat should be < 1.01)
        rhat_values = az.rhat(result)
        max_rhat = max([float(rhat_values[var].max()) for var in rhat_values.data_vars])
        print(f"\nMax R-hat: {max_rhat:.4f} {'✓' if max_rhat < 1.01 else '⚠ Warning: convergence issues'}")
        
        return model, result
    
    def poststratify(
        self, 
        response_col: str, 
        poststrat_table: pd.DataFrame,
        response_type: str = "continuous"
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Apply poststratification to get population-level estimates
        
        Parameters
        ----------
        response_col : str
            Name of response variable
        poststrat_table : pd.DataFrame
            Poststratification table with demographic combinations and weights
        response_type : str
            Type of response variable: "continuous", "binary", "count", or "ordinal"
            
        Returns
        -------
        poststrat_estimates : pd.DataFrame
            Estimates for each demographic subgroup
        aggregate_stats : Dict[str, float]
            Population-level aggregate statistics
        """
        print(f"\nPoststratifying {response_col}...")
        
        if response_col not in self.models:
            raise ValueError(f"Model for {response_col} not fitted. Call fit_multilevel_model() first.")
        
        model = self.models[response_col]
        result = self.results[response_col]
        
        # Prepare poststratification data (need same structure as fitting data)
        pred_data = poststrat_table.copy()
        for col in self.demographic_cols:
            pred_data[col] = pd.Categorical(
                pred_data[col],
                categories=self.demographics_schema[col]
            )
        
        # Generate predictions for each stratum
        print("Generating predictions for all strata...")
        predictions = model.predict(
            result, 
            data=pred_data, 
            kind="response",
            inplace=False
        )
        
        # Extract posterior predictive samples
        # For different families, the variable name may vary
        try:
            posterior_pred = az.extract(
                predictions, 
                group="posterior_predictive",
                var_names=[response_col]
            )[response_col]
        except KeyError:
            # Try alternative names (e.g., 'mu' for some families)
            pred_vars = list(predictions.posterior_predictive.data_vars)
            if len(pred_vars) > 0:
                posterior_pred = az.extract(
                    predictions,
                    group="posterior_predictive"
                )[pred_vars[0]]
            else:
                raise ValueError(f"Could not find posterior predictions for {response_col}")
        
        # Calculate point estimates and credible intervals for each stratum
        pred_data['estimate_mean'] = posterior_pred.mean(dim='sample').values
        pred_data['estimate_median'] = posterior_pred.median(dim='sample').values
        pred_data['estimate_std'] = posterior_pred.std(dim='sample').values
        pred_data['ci_lower'] = posterior_pred.quantile(0.025, dim='sample').values
        pred_data['ci_upper'] = posterior_pred.quantile(0.975, dim='sample').values
        
        # Poststratify: weight each stratum's estimate by population proportion
        # Following MRP formula: E[Y] = sum_j ( p_j * E[Y_j] )
        # where p_j is proportion of population in stratum j
        
        weighted_samples = posterior_pred.values * pred_data['proportion'].values[:, np.newaxis]
        aggregate_mean = np.sum(weighted_samples, axis=0).mean()
        aggregate_median = np.median(np.sum(weighted_samples, axis=0))
        aggregate_std = np.sum(weighted_samples, axis=0).std()
        aggregate_ci_lower = np.percentile(np.sum(weighted_samples, axis=0), 2.5)
        aggregate_ci_upper = np.percentile(np.sum(weighted_samples, axis=0), 97.5)
        
        aggregate_stats = {
            'response_variable': response_col,
            'response_type': response_type,
            'estimate_mean': float(aggregate_mean),
            'estimate_median': float(aggregate_median),
            'estimate_std': float(aggregate_std),
            'ci_lower': float(aggregate_ci_lower),
            'ci_upper': float(aggregate_ci_upper),
            'total_population': int(pred_data['population_count'].sum())
        }
        
        print(f"\nPopulation-level estimate for {response_col}:")
        print(f"  Mean: {aggregate_mean:.4f} (95% CI: [{aggregate_ci_lower:.4f}, {aggregate_ci_upper:.4f}])")
        
        return pred_data, aggregate_stats
    
    def run_full_analysis(
        self,
        response_types: Dict[str, str],
        population_weights: Optional[pd.DataFrame] = None,
        draws: int = 2000,
        target_accept: float = 0.95,
        random_seed: int = 42
    ):
        """
        Run complete MrP analysis for all response variables
        
        Parameters
        ----------
        response_types : Dict[str, str]
            Dictionary mapping response column names to their types
            ("continuous", "binary", or "count")
        population_weights : pd.DataFrame, optional
            Population weights for poststratification
        draws : int
            Number of MCMC samples
        target_accept : float
            Target acceptance rate
        random_seed : int
            Random seed
        """
        print("\n" + "="*70)
        print("MULTILEVEL REGRESSION AND POSTSTRATIFICATION (MrP) ANALYSIS")
        print("="*70)
        
        # Create poststratification table
        poststrat_table = self.prepare_poststratification_table(population_weights)
        
        # Save poststratification table
        poststrat_path = self.output_dir / "poststratification_table.csv"
        poststrat_table.to_csv(poststrat_path, index=False)
        print(f"\n✓ Saved poststratification table to {poststrat_path}")
        
        all_subgroup_estimates = {}
        all_aggregate_estimates = []
        
        # Process each response variable
        for response_col in self.response_cols:
            if response_col not in response_types:
                print(f"\n⚠ Skipping {response_col}: response type not specified")
                continue
            
            response_type = response_types[response_col]
            
            # Fit model
            self.fit_multilevel_model(
                response_col,
                response_type=response_type,
                draws=draws,
                target_accept=target_accept,
                random_seed=random_seed
            )
            
            # Check if model was fitted (may return None if insufficient data)
            if response_col not in self.models:
                print(f"  ⚠ Skipped {response_col} due to insufficient data\n")
                continue
            
            # Poststratify
            subgroup_estimates, aggregate_stats = self.poststratify(
                response_col,
                poststrat_table,
                response_type=response_type
            )
            
            all_subgroup_estimates[response_col] = subgroup_estimates
            all_aggregate_estimates.append(aggregate_stats)
        
        # Save results
        self._save_results(all_subgroup_estimates, all_aggregate_estimates)
        
        print("\n" + "="*70)
        print("MrP ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}")
    
    def _save_results(
        self,
        subgroup_estimates: Dict[str, pd.DataFrame],
        aggregate_estimates: List[Dict]
    ):
        """Save analysis results to files"""
        
        # Save subgroup estimates for each response variable
        for response_col, estimates_df in subgroup_estimates.items():
            output_path = self.output_dir / f"subgroup_estimates_{response_col}.csv"
            estimates_df.to_csv(output_path, index=False)
            print(f"✓ Saved subgroup estimates for {response_col} to {output_path}")
        
        # Save aggregate estimates
        aggregate_df = pd.DataFrame(aggregate_estimates)
        aggregate_path = self.output_dir / "aggregate_estimates.csv"
        aggregate_df.to_csv(aggregate_path, index=False)
        print(f"✓ Saved aggregate estimates to {aggregate_path}")
        
        # Save aggregate estimates as JSON for easy reading
        aggregate_json_path = self.output_dir / "aggregate_estimates.json"
        with open(aggregate_json_path, 'w') as f:
            json.dump(aggregate_estimates, f, indent=2)
        print(f"✓ Saved aggregate estimates to {aggregate_json_path}")


def main():
    """Command-line interface for MrP analysis"""
    parser = argparse.ArgumentParser(
        description="Multilevel Regression and Poststratification (MrP) Analysis"
    )
    parser.add_argument(
        "data_path",
        help="Path to CSV file with individual-level survey data"
    )
    parser.add_argument(
        "demographics_schema",
        help="Path to JSON file specifying valid demographic categories"
    )
    parser.add_argument(
        "response_types",
        help="Path to JSON file mapping response variables to types (continuous/binary/count)"
    )
    parser.add_argument(
        "--population-weights",
        help="Path to CSV file with population weights for each demographic combination",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save output files",
        default="mrp_results"
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Number of MCMC samples to draw (default: 2000)"
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
        help="Target acceptance rate for MCMC (default: 0.95)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load response types
    with open(args.response_types, 'r') as f:
        response_types = json.load(f)
    
    # Load population weights if provided
    population_weights = None
    if args.population_weights:
        population_weights = pd.read_csv(args.population_weights)
    
    # Initialize analyzer
    analyzer = MrPAnalyzer(
        args.data_path,
        args.demographics_schema,
        output_dir=args.output_dir
    )
    
    # Run analysis
    analyzer.run_full_analysis(
        response_types=response_types,
        population_weights=population_weights,
        draws=args.draws,
        target_accept=args.target_accept,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
