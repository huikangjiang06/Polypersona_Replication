# Multilevel Regression and Poststratification (MrP)

This directory contains an implementation of **Multilevel Regression and Poststratification (MrP)**, a statistical method for adjusting non-representative survey samples using demographic information and population weights.

## Key References

- **Park, Gelman & Bafumi (2004)**: "Bayesian Multilevel Estimation with Poststratification"
- **Gelman & Little (1997)**: "Poststratification into many categories using hierarchical logistic regression"
- **Bambi Tutorial**: https://bambinos.github.io/bambi/notebooks/mister_p.html
- **Stan User Guide**: https://mc-stan.org/docs/stan-users-guide/poststratification.html

## Installation

Install required dependencies:

```bash
pip install bambi arviz pandas numpy matplotlib seaborn
```

**Note**: Bambi requires PyMC and PyTensor. On some systems you may need:
```bash
pip install pymc>=5.0 pytensor>=2.0
```

## Quick Start

### 1. Generate Test Data

```bash
python mrp_utils.py
```

This creates example files in `example_data/`:
- `survey_data.csv` - Simulated survey responses
- `demographics_schema.json` - Valid demographic categories
- `response_types.json` - Response variable types
- `population_weights.csv` - Population counts per demographic stratum

### 2. Run MrP Analysis

```bash
python mrp_analysis.py \
    example_data/survey_data.csv \
    example_data/demographics_schema.json \
    example_data/response_types.json \
    --population-weights example_data/population_weights.csv \
    --output-dir mrp_results
```

### 3. View Results

Results are saved in `mrp_results/`:
- `aggregate_estimates.csv` - Population-level estimates
- `subgroup_estimates_<variable>.csv` - Estimates for each demographic subgroup
- `poststratification_table.csv` - Full demographic stratification table

## Input Data Format

### 1. Survey Data CSV

Individual-level survey data with demographics and responses:

```csv
age_bracket,income_bracket,education,region,approval,satisfaction
18-29,30-60k,bachelors,west,1,5
45-59,60-100k,graduate,northeast,0,6
...
```

**Structure**:
- First N columns: Demographics (categorical variables)
- Remaining M columns: Survey responses (numerical values)

### 2. Demographics Schema JSON

Defines all valid demographic categories:

```json
{
  "age_bracket": ["18-29", "30-44", "45-59", "60+"],
  "income_bracket": ["<30k", "30-60k", "60-100k", "100k+"],
  "education": ["high_school", "some_college", "bachelors", "graduate"],
  "region": ["northeast", "south", "midwest", "west"]
}
```

**Note**: All demographic values in the survey data must match these categories exactly.

### 3. Response Types JSON

Maps response variables to their statistical types:

```json
{
  "approval": "binary",
  "satisfaction": "count",
  "income_score": "continuous"
}
```

**Supported types**:
- `"binary"` - Binary outcomes (0/1) → Bernoulli/logistic regression
- `"count"` - Count data (0, 1, 2, ...), use this category for likert/ordinal data → Poisson regression  
- `"continuous"` - Real-valued outcomes → Gaussian regression

### 4. Population Weights CSV (Optional)

Population counts for each demographic combination:

```csv
age_bracket,income_bracket,education,region,population_count
18-29,<30k,high_school,northeast,125000
18-29,<30k,high_school,south,185000
...
```

**Note**: If not provided, sample proportions are used as proxy weights.

## Usage Details

### Command-Line Interface

```bash
python mrp_analysis.py <data_path> <demographics_schema> <response_types> [OPTIONS]
```

**Required Arguments**:
- `data_path` - Path to survey data CSV
- `demographics_schema` - Path to demographics schema JSON
- `response_types` - Path to response types JSON

**Optional Arguments**:
- `--population-weights` - Path to population weights CSV
- `--output-dir` - Output directory (default: `mrp_results`)
- `--draws` - Number of MCMC samples, hyperparameter (default: 2000)
- `--target-accept` - MCMC acceptance rate (default: 0.95)
- `--random-seed` - Random seed for reproducibility (default: 42)

### Python API

```python
from mrp_analysis import MrPAnalyzer

# Initialize analyzer
analyzer = MrPAnalyzer(
    data_path="survey_data.csv",
    demographics_schema_path="demographics_schema.json",
    output_dir="mrp_results"
)

# Prepare poststratification table
poststrat_table = analyzer.prepare_poststratification_table(
    population_weights=population_weights_df  # Optional
)

# Fit model for a single response variable
model, result = analyzer.fit_multilevel_model(
    response_col="approval",
    response_type="binary",
    draws=2000
)

# Poststratify to get population estimates
subgroup_estimates, aggregate_stats = analyzer.poststratify(
    response_col="approval",
    poststrat_table=poststrat_table,
    response_type="binary"
)

# Or run complete analysis for all responses
response_types = {"approval": "binary", "satisfaction": "count"}
analyzer.run_full_analysis(
    response_types=response_types,
    population_weights=population_weights_df
)
```

## Output Files

### 1. `aggregate_estimates.csv`

Population-level estimates for each response variable:

| response_variable | response_type | estimate_mean | estimate_median | estimate_std | ci_lower | ci_upper | total_population |
|-------------------|---------------|---------------|-----------------|--------------|----------|----------|------------------|
| approval | binary | 0.5234 | 0.5241 | 0.0156 | 0.4928 | 0.5537 | 10000000 |
| satisfaction | count | 4.6782 | 4.6801 | 0.0892 | 4.5032 | 4.8521 | 10000000 |

**Columns**:
- `estimate_mean` - Posterior mean of population estimate
- `estimate_median` - Posterior median
- `estimate_std` - Posterior standard deviation
- `ci_lower`, `ci_upper` - 95% credible interval bounds

### 2. `subgroup_estimates_<variable>.csv`

Estimates for each demographic subgroup:

| age_bracket | income_bracket | education | region | population_count | proportion | estimate_mean | estimate_median | estimate_std | ci_lower | ci_upper |
|-------------|----------------|-----------|--------|------------------|------------|---------------|-----------------|--------------|----------|----------|
| 18-29 | <30k | high_school | northeast | 125000 | 0.0125 | 0.4523 | 0.4519 | 0.0234 | 0.4062 | 0.4981 |
| 18-29 | <30k | high_school | south | 185000 | 0.0185 | 0.5012 | 0.5008 | 0.0198 | 0.4623 | 0.5397 |

**Columns**:
- Demographics columns - Identifying the subgroup
- `population_count` - Number of individuals in this stratum
- `proportion` - Proportion of total population in this stratum
- `estimate_mean`, etc. - Same as aggregate estimates, but for this subgroup

### 3. `poststratification_table.csv`

Complete table of all demographic combinations with population weights.

## Utility Functions

Additional utilities in `mrp_utils.py`:

### Generate Example Data
```python
from mrp_utils import create_example_data

paths = create_example_data(
    n_samples=1000,
    output_dir="example_data"
)
```

### Visualize Subgroup Estimates
```python
from mrp_utils import plot_subgroup_estimates

plot_subgroup_estimates(
    estimates_path="mrp_results/subgroup_estimates_approval.csv",
    demographic_var="age_bracket",
    output_path="age_estimates.png"
)
```

### Compare Naive vs MrP Estimates
```python
from mrp_utils import compare_naive_vs_mrp

compare_naive_vs_mrp(
    data_path="survey_data.csv",
    mrp_results_path="mrp_results/aggregate_estimates.csv",
    response_var="approval",
    output_path="comparison.png"
)
```

### Generate Summary Report
```python
from mrp_utils import create_summary_report

create_summary_report(
    results_dir="mrp_results",
    output_path="summary_report.md"
)
```