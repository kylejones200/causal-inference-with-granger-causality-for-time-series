# Causal Inference with Granger Causality for Time Series

*Moving from correlation to causation — carefully*

---

Correlation is easy to find in time series data. Two variables that both trend upward will correlate. Two variables driven by the same seasonal cycle will correlate. None of that tells you anything about causation. Causal inference methods try to answer a harder question: if you intervened on variable X, what would happen to variable Y?

For time series, the toolbox includes Granger causality, structural VAR models, local projections, and the synthetic control method. Each makes different assumptions and answers a slightly different question. None of them guarantee true causation — that requires a credible identification argument, not just a statistical test.

## Granger Causality: Does X Help Predict Y?

Granger causality tests whether past values of X improve predictions of Y, over and above Y's own past values. It is not causation in the philosophical sense — it is predictive precedence. But in many domains, predictive precedence is useful evidence.

```python
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def test_granger(data, var1, var2, max_lags=12):
    xy = pd.concat([data[var1], data[var2]], axis=1)
    results = grangercausalitytests(xy, maxlag=max_lags, verbose=False)
    
    summary = pd.DataFrame(index=range(1, max_lags + 1), columns=['F-stat', 'p-value'])
    for lag in range(1, max_lags + 1):
        summary.loc[lag] = [
            results[lag][0]['ssr_ftest'][0],
            results[lag][0]['ssr_ftest'][1]
        ]
    return summary
```

The test fits two models: a restricted model using only Y's own lags, and an unrestricted model using both Y's and X's lags. If the F-test rejects the restriction, X Granger-causes Y at that lag length.

A few things to watch:
- **Stationarity required.** Both series must be stationary before testing. Apply ADF or KPSS tests first and difference if needed.
- **Lag selection.** Use AIC or BIC to select the lag length, or test across a range and look for stability in the p-values.
- **Reverse causality.** Always test both directions. If X Granger-causes Y and Y Granger-causes X, you have a feedback loop, not a unidirectional causal chain.

Granger causality is most useful as an exploratory tool — it tells you which variable relationships are worth investigating more carefully with a structural model.

## Structural Vector Autoregression (SVAR)

VAR models treat multiple time series as mutually dependent — each variable is regressed on its own lags and the lags of all other variables in the system. SVARs add economic structure to the VAR by imposing restrictions that reflect theoretical beliefs about which contemporaneous relationships are plausible.

```python
from statsmodels.tsa.api import VAR

var_model = VAR(data)
var_result = var_model.fit(maxlags=4, ic='aic')
print(var_result.summary())

irf = var_result.irf(periods=20)
irf.plot(orth=True)
```

The impulse response function (IRF) traces how a one-unit shock to one variable propagates through the system over time. Orthogonalized IRFs (Cholesky decomposition) assume a specific causal ordering — the ordering of variables in your dataset determines which contemporaneous shocks are allowed. This is a strong assumption and should be justified on domain grounds.

Forecast error variance decomposition (FEVD) shows what fraction of the forecast variance of each variable is attributable to shocks from each other variable — useful for understanding which relationships dominate the system at different horizons.

## Local Projections

Local projections estimate impulse responses directly using a sequence of regressions rather than inverting a VAR. For each horizon h, regress the outcome at time t+h on the shock at time t and controls:

```python
import statsmodels.api as sm
import numpy as np

def local_projections(data, dep_var, shock_var, controls=None, horizons=20):
    responses = []
    cis = []
    for h in range(horizons + 1):
        y = data[dep_var].shift(-h)
        X = data[[shock_var]]
        if controls:
            X = pd.concat([X, data[controls]], axis=1)
        X = sm.add_constant(X)
        valid = y.notna()
        result = sm.OLS(y[valid], X[valid]).fit(
            cov_type='HAC', cov_kwds={'maxlags': h + 1}
        )
        responses.append(result.params[shock_var])
        cis.append(result.conf_int().loc[shock_var].values)
    return np.array(responses), np.array(cis)
```

Local projections are more flexible than VAR-based IRFs — they do not impose the VAR's parametric structure on the impulse response shape. They tend to produce wider confidence intervals but are more robust to model misspecification. For policy analysis at long horizons (beyond 8–12 periods), local projections are generally preferred.

## Synthetic Control Method

The synthetic control method is for cases where you have one treated unit (a country, state, or company) and a set of untreated comparators. It constructs a weighted average of the control units that best matches the treated unit in the pre-treatment period, then uses that synthetic control as the counterfactual post-treatment.

```python
from scipy.optimize import minimize

def build_synthetic_control(pre_treated, pre_controls):
    n_controls = pre_controls.shape[1]
    
    def objective(weights):
        synthetic = pre_controls @ weights
        return np.mean((pre_treated - synthetic) ** 2)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
    ]
    bounds = [(0, 1)] * n_controls
    
    result = minimize(
        objective,
        x0=np.ones(n_controls) / n_controls,
        constraints=constraints,
        bounds=bounds
    )
    return result.x
```

The treatment effect at each post-treatment period is the gap between the treated unit's actual outcome and the synthetic control's outcome. The method works well when the pre-treatment fit is good — a poor pre-treatment match means the counterfactual is not credible.

Inference is done through placebo tests: run the same procedure on each control unit as if it had been treated, and compare the treated unit's post-treatment gap to the distribution of placebo gaps.

## What These Methods Cannot Do

None of these methods prove causation. They all require assumptions that data cannot verify:

- **Granger causality** assumes no omitted confounders that cause both X and Y. If a third variable drives both, X will appear to Granger-cause Y.
- **SVAR** requires you to impose a structural ordering or other restrictions. The results are only as credible as those assumptions.
- **Local projections** require the shock to be exogenous — not caused by the outcome variable contemporaneously.
- **Synthetic control** requires the parallel trends assumption: without treatment, the treated unit would have evolved like the synthetic control.

The credibility of a causal claim rests on the identification argument, not the statistical machinery. The statistics are only as good as the economic reasoning behind them.

## Key Takeaways

- Granger causality tests predictive precedence, not true causation — useful as an exploratory screen, not a final answer.
- SVAR models impose economic structure on the causal ordering and trace shocks through impulse response functions.
- Local projections estimate IRFs more robustly than VAR at long horizons, at the cost of wider confidence intervals.
- Synthetic control builds a data-driven counterfactual from a weighted combination of untreated units.
- All causal methods require identification assumptions that must be justified on domain grounds, not validated statistically.
