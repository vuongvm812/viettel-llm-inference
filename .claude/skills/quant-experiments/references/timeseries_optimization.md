# Time Series & Optimization — Patterns Reference

## Time Series Model Selection

### Decision Tree

```
Is the series stationary (ADF p < 0.05)?
│
├── NO → Difference the series (d=1 or d=2); test again
│
└── YES → Inspect ACF / PACF
    ├── ACF tails off + PACF cuts off at lag p → AR(p)
    ├── ACF cuts off at lag q + PACF tails off → MA(q)
    ├── Both tail off → ARMA(p,q)
    └── Strong seasonal pattern → SARIMA or ETS
```

### Model Library Quick Reference

| Model | Use case | Library |
|-------|----------|---------|
| ARIMA | Univariate, stationary after differencing | `statsmodels.tsa.arima.model.ARIMA` |
| SARIMA | Seasonal ARIMA | `statsmodels.tsa.statespace.sarimax.SARIMAX` |
| ETS (Holt-Winters) | Exponential smoothing with trend+season | `statsmodels.tsa.holtwinters.ExponentialSmoothing` |
| Prophet | Daily/weekly/yearly seasonality, holidays | `prophet.Prophet` |
| LSTM | Non-linear, multivariate | `tensorflow.keras` |
| sktime | Unified interface for many models | `sktime` |

### Stationarity Tests
```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF: H0 = unit root (non-stationary); reject → stationary
adf_stat, adf_p, _, _, crit, _ = adfuller(series, autolag='AIC')
print(f"ADF stat={adf_stat:.3f}, p={adf_p:.4f} → {'Stationary' if adf_p<0.05 else 'Non-stationary'}")

# KPSS: H0 = stationary; reject → non-stationary (use both for robustness)
kpss_stat, kpss_p, _, crit = kpss(series, regression='c')
print(f"KPSS stat={kpss_stat:.3f}, p={kpss_p:.4f}")
```

### ACF / PACF Plots
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(series, lags=40, ax=ax1, title='ACF — guides MA order (q)')
plot_pacf(series, lags=40, ax=ax2, title='PACF — guides AR order (p)')
plt.tight_layout()
```

### Forecast Evaluation
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def forecast_metrics(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    da   = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100
    return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'Dir. Acc. (%)': da})
```

---

## Numerical Optimization Patterns

### Algorithm Selection Guide

```
Is the objective differentiable?
│
├── YES
│   ├── Convex → L-BFGS-B, SLSQP (scipy.optimize.minimize)
│   ├── Non-convex (few local minima) → L-BFGS-B with multiple restarts
│   └── Non-convex (many local minima) → Differential Evolution or Basin-hopping
│
└── NO (black-box / noisy)
    ├── Low-dim (<20) → Nelder-Mead, Powell
    ├── Medium-dim → CMA-ES (pip install cma)
    └── Expensive evaluations → Bayesian Optimization (scikit-optimize, optuna)
```

### Multiple Restarts (for non-convex)
```python
best_result = None
all_results = []

for trial in range(N_RESTARTS):
    x0 = RNG.uniform(low, high, size=n_params)
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    all_results.append({'trial': trial, 'f(x*)': res.fun, 'x*': res.x})
    if best_result is None or res.fun < best_result.fun:
        best_result = res

restarts_df = pd.DataFrame(all_results).drop('x*', axis=1)
```

### Differential Evolution (global, gradient-free)
```python
from scipy.optimize import differential_evolution

result = differential_evolution(
    objective,
    bounds=list(zip(lower_bounds, upper_bounds)),
    seed=SEED,
    maxiter=1000,
    tol=1e-7,
    popsize=15,
    mutation=(0.5, 1.0),
    recombination=0.7,
    callback=lambda xk, convergence: history.append(objective(xk)),
    workers=1  # set to -1 for parallel
)
```

### Bayesian Optimization (expensive objectives)
```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_objective

space = [Real(0.01, 1.0, name='learning_rate'),
         Integer(50, 500, name='n_estimators')]

result = gp_minimize(objective, space, n_calls=50, random_state=SEED,
                     acq_func='EI', n_initial_points=10)

plot_convergence(result)
```

### Constraint Handling
```python
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

# Linear constraint: A @ x <= b
linear_constraint = LinearConstraint(A_matrix, lb=-np.inf, ub=b_vector)

# Nonlinear constraint: g(x) >= 0
nonlinear_constraint = NonlinearConstraint(g_func, lb=0, ub=np.inf)

result = minimize(objective, x0, method='SLSQP',
                  constraints=[linear_constraint, nonlinear_constraint],
                  bounds=bounds)
```
