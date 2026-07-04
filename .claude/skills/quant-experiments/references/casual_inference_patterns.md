# Causal Inference Patterns

## Choosing a Design

```
Do you have randomization?
│
├── YES → Randomized Controlled Trial (RCT)
│   ├── Simple A/B → t-test or chi-square
│   ├── Multi-arm → ANOVA + pairwise with correction
│   └── Blocked/stratified → Mixed-effects model (pingouin or statsmodels)
│
└── NO → Observational Study
    ├── Pre/post with control group → Difference-in-Differences (DiD)
    ├── Treatment has a threshold → Regression Discontinuity (RD)
    ├── Instrumental variable available → IV / 2SLS
    └── No natural experiment → Propensity Score Methods
        ├── Matching → causalinference or sklearn NearestNeighbors
        └── Weighting (IPW) → manual or dowhy
```

---

## A/B Test (RCT) — Standard Pattern

```python
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower

# --- Pre-experiment: sample size ---
effect_size = 0.05  # MDE as proportion
baseline = 0.10
alpha = 0.05
power = 0.80

analysis = NormalIndPower()
n_per_group = analysis.solve_power(
    effect_size=effect_size / np.sqrt(baseline * (1 - baseline)),
    alpha=alpha, power=power, alternative='two-sided'
)
print(f"Required n per group: {int(np.ceil(n_per_group))}")

# --- Analysis ---
counts = np.array([conversions_treatment, conversions_control])
nobs = np.array([n_treatment, n_control])
z_stat, p_value = proportions_ztest(counts, nobs)
```

---

## Difference-in-Differences

```python
import statsmodels.formula.api as smf

# Data must have: unit_id, time (pre=0, post=1), treated (0/1), outcome
did_model = smf.ols(
    'outcome ~ time + treated + time:treated',
    data=panel_df
).fit(cov_type='HC3')  # Heteroskedasticity-robust SEs

print(did_model.summary())
# Coefficient on time:treated is the DiD estimator (ATT)
```

**Key assumptions to state:**
- Parallel trends in pre-period (plot and test visually)
- No anticipation effects
- SUTVA (no spillovers)

---

## Propensity Score Matching

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# 1. Estimate propensity scores
X_covariates = df[covariate_cols]
lr = LogisticRegression(max_iter=1000)
lr.fit(X_covariates, df['treated'])
df['pscore'] = lr.predict_proba(X_covariates)[:, 1]

# 2. Match treated to control (1:1 nearest neighbor)
treated = df[df['treated'] == 1]
control = df[df['treated'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])

matched_control = control.iloc[indices.flatten()]
matched_df = pd.concat([treated, matched_control])

# 3. Check covariate balance (Standardized Mean Difference)
def smd(col, df):
    t = df[df['treated']==1][col]
    c = df[df['treated']==0][col]
    return (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)

balance = pd.DataFrame({
    'Before': [smd(c, df) for c in covariate_cols],
    'After':  [smd(c, matched_df) for c in covariate_cols]
}, index=covariate_cols)
# SMD < 0.1 is generally considered good balance
```

---

## Bayesian A/B (Beta-Binomial)

```python
# No pymc required — analytical solution for proportions
from scipy.stats import beta

# Priors: Beta(1,1) = uniform
alpha_a, beta_a = 1 + conversions_a, 1 + (n_a - conversions_a)
alpha_b, beta_b = 1 + conversions_b, 1 + (n_b - conversions_b)

# Posterior samples
samples_a = beta.rvs(alpha_a, beta_a, size=100_000)
samples_b = beta.rvs(alpha_b, beta_b, size=100_000)

prob_b_better = (samples_b > samples_a).mean()
expected_lift = (samples_b - samples_a).mean()
ci_lift = np.percentile(samples_b - samples_a, [2.5, 97.5])

print(f"P(B > A) = {prob_b_better:.3f}")
print(f"Expected lift = {expected_lift:.4f}, 95% CrI {ci_lift}")
```
