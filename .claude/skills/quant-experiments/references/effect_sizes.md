# Effect Sizes — Formulas & Interpretation

Effect sizes are mandatory in publication-ready reports. Always report alongside p-values.

## By Test Type

### T-tests → Cohen's d

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}$$

```python
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s_pooled = np.sqrt(((n1-1)*x1.std(ddof=1)**2 + (n2-1)*x2.std(ddof=1)**2) / (n1+n2-2))
    return (x1.mean() - x2.mean()) / s_pooled
```

| d | Interpretation |
|---|---------------|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### ANOVA → Eta-squared (η²) and Partial η²

$$\eta^2 = \frac{SS_{between}}{SS_{total}}$$

```python
# Via pingouin (preferred)
import pingouin as pg
pg.anova(data=df, dv='outcome', between='group', detailed=True)
# Returns eta-squared automatically
```

| η² | Interpretation |
|----|---------------|
| 0.01 | Small |
| 0.06 | Medium |
| 0.14 | Large |

### Chi-square → Cramér's V

$$V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}}$$

```python
def cramers_v(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, c = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))
```

| V | Interpretation |
|---|---------------|
| 0.1 | Small |
| 0.3 | Medium |
| 0.5 | Large |

### Correlation → r (Pearson/Spearman)

| r | Interpretation |
|---|---------------|
| 0.1 | Small |
| 0.3 | Medium |
| 0.5 | Large |

### Mann-Whitney → r (rank-biserial)

$$r = \frac{U}{n_1 \cdot n_2} \cdot 2 - 1$$

```python
def rank_biserial(x1, x2):
    u_stat = stats.mannwhitneyu(x1, x2).statistic
    return (2 * u_stat) / (len(x1) * len(x2)) - 1
```

## Confidence Intervals for Effect Sizes

```python
import pingouin as pg

# Cohen's d with 95% CI
result = pg.ttest(x1, x2)
d = result['cohen-d'].values[0]
ci = result['CI95%'].values[0]  # [low, high]
print(f"d = {d:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Reporting Template (APA Style)

```python
def apa_report_ttest(t, df, p, d, ci_low, ci_high):
    p_str = f"= {p:.3f}" if p >= 0.001 else "< .001"
    return (f"t({df:.0f}) = {t:.2f}, p {p_str}, "
            f"d = {d:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]")

def apa_report_chisq(chi2, df, p, v):
    p_str = f"= {p:.3f}" if p >= 0.001 else "< .001"
    return f"χ²({df}) = {chi2:.2f}, p {p_str}, V = {v:.2f}"
```
