# Statistical Tests Decision Guide

## Test Selection Decision Tree

### Comparing Groups

```
Is the outcome variable continuous or categorical?
│
├── Continuous
│   ├── 2 groups
│   │   ├── Independent → Welch's t-test (scipy.stats.ttest_ind, equal_var=False)
│   │   │   └── Non-normal or small n → Mann-Whitney U (scipy.stats.mannwhitneyu)
│   │   └── Paired / repeated → Paired t-test (scipy.stats.ttest_rel)
│   │       └── Non-normal → Wilcoxon signed-rank (scipy.stats.wilcoxon)
│   └── 3+ groups
│       ├── Independent → One-way ANOVA (scipy.stats.f_oneway)
│       │   └── Non-normal → Kruskal-Wallis (scipy.stats.kruskal)
│       └── Repeated measures → Use pingouin.rm_anova
│
└── Categorical
    ├── 2×2 contingency → Fisher's exact (n<1000) or Chi-square
    ├── R×C contingency → Chi-square (scipy.stats.chi2_contingency)
    └── Proportions (2 group) → Z-test (statsmodels.stats.proportion.proportions_ztest)
```

### Correlation & Association

| Goal | Test | Function |
|------|------|----------|
| Linear association (normal) | Pearson r | `scipy.stats.pearsonr` |
| Monotonic association | Spearman ρ | `scipy.stats.spearmanr` |
| Concordance (ordinal) | Kendall τ | `scipy.stats.kendalltau` |
| Multivariate correlation | Multiple regression | `statsmodels.OLS` |

### Goodness of Fit

| Goal | Test | Function |
|------|------|----------|
| Does data follow distribution X? | KS test | `scipy.stats.kstest` |
| Is sample from same dist as ref? | 2-sample KS | `scipy.stats.ks_2samp` |
| Normality (n < 50) | Shapiro-Wilk | `scipy.stats.shapiro` |
| Normality (n ≥ 50) | D'Agostino K² | `scipy.stats.normaltest` |

---

## Assumptions Checklist by Test

### Welch's t-test
- [ ] Independence of observations
- [ ] Approximately normal OR n > 30 per group (CLT)
- [ ] No extreme outliers

### ANOVA
- [ ] Independence
- [ ] Normality within groups (check with Shapiro per group)
- [ ] Homogeneity of variance → Levene's test (`scipy.stats.levene`)

### Chi-square
- [ ] Expected cell frequency ≥ 5 in all cells
- [ ] Independent observations
- [ ] If any expected < 5 → use Fisher's exact

### Mann-Whitney U
- [ ] Independence
- [ ] Ordinal or continuous outcome
- [ ] Tests median equality IF distributions have same shape; otherwise tests stochastic dominance

---

## Multiple Comparisons

Whenever running >1 test on the same dataset, apply correction:

```python
from statsmodels.stats.multitest import multipletests

# Recommended: Benjamini-Hochberg FDR (less conservative than Bonferroni)
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Bonferroni (most conservative, use when controlling FWER strictly)
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
```

Always report both raw and corrected p-values in the results table.
