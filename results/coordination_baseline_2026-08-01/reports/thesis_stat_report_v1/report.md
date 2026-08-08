# Statistical Report

- Results root: /workspace/results/coordination_baseline_2026-08-01/experiments
- Conditions: autogen, crewai, langgraph
- Observations: 24

## Summary Statistics

framework  n  success_rate_mean  success_rate_std  success_rate_ci_low  success_rate_ci_high  latency_seconds_mean  latency_seconds_std  latency_seconds_ci_low  latency_seconds_ci_high  token_cost_usd_mean  token_cost_usd_std  token_cost_usd_ci_low  token_cost_usd_ci_high
  autogen  8               1.00           0.00000             1.000000              1.000000              0.001823             0.004275               -0.001751                 0.005397                  0.0                 0.0                    0.0                     0.0
   crewai  8               1.00           0.00000             1.000000              1.000000             15.406592            10.122754                6.943758                23.869426                  0.0                 0.0                    0.0                     0.0
langgraph  8               0.75           0.46291             0.362998              1.137002              1.975371             1.348863                0.847693                 3.103048                  0.0                 0.0                    0.0                     0.0

## Pairwise Comparison Files

- success: pairwise_success.csv
- latency_seconds: pairwise_latency_seconds.csv
- cost_usd: pairwise_cost_usd.csv

## Advanced Statistics Files

- descriptive_bootstrap: descriptive_bootstrap.csv
- success_rate_tests: success_rate_tests.csv
- mast_distribution_test: mast_distribution_test.csv
- continuous_latency_seconds_tests: continuous_latency_seconds_tests.csv
- continuous_cost_usd_tests: continuous_cost_usd_tests.csv

## ANOVA

### success

- metric: success
- condition_column: framework
- n_groups: 3
- n_observations: 24
- f_statistic: 2.333333333333334
- p_value: 0.1215970824980537
- eta_squared: 0.18181818181818182

### latency_seconds

- metric: latency_seconds
- condition_column: framework
- n_groups: 3
- n_observations: 24
- f_statistic: 16.170338707389913
- p_value: 5.6129097979689036e-05
- eta_squared: 0.6063042125111586

### cost_usd

- metric: cost_usd
- condition_column: framework
- n_groups: 3
- n_observations: 24
- f_statistic: nan
- p_value: nan
- eta_squared: nan

## Figures

- figures/success_rate_grouped.png
- figures/success_rate_grouped.pdf
- figures/mast_categories_grouped.png
- figures/mast_categories_grouped.pdf
- figures/latency_distribution.png
- figures/latency_distribution.pdf
- figures/cost_distribution.png
- figures/cost_distribution.pdf