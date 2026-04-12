# Causal Inference

Understanding causal relationships in economic time series goes beyond simple correlation analysis. This chapter explores methods for identifying and quantifying causal relationships in temporal economic data, from Granger causality to modern causal inference techniques.

## Granger Causality: The Foundation

Granger causality is a fundamental tool for assessing whether one time series can predict another.

    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.api import VAR

    class GrangerAnalysis:
        def __init__(self, data):
            self.data = data
        def test_granger_causality(self, variable1, variable2, max_lags=12):
            """Test for Granger causality between two variables"""
            data = pd.concat([self.data[variable1], self.data[variable2]], axis=1)
            results = grangercausalitytests(data, maxlag=max_lags)
            causality_results = pd.DataFrame(
                index=range(1, max_lags + 1),
                columns=['F-statistic', 'p-value']
            )
            for lag in range(1, max_lags + 1):
                causality_results.loc[lag] = [
                    results[lag][0]['ssr_ftest'][0],
                    results[lag][0]['ssr_ftest'][1]
                ]
            return causality_results
        def plot_causality_results(self, results):
            """Plot p-values for different lag orders"""
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(results.index, results['p-value'], marker='o')
            plt.axhline(y=0.05, color='r', linestyle='--', label='5% significance')
            plt.xlabel('Lag Order')
            plt.ylabel('p-value')
            plt.title('Granger Causality Test Results')
            plt.legend()
            plt.show()

## Structural Vector Autoregression (SVAR)

SVAR models extend the Vector Autoregression (VAR) framework by incorporating structural restrictions.

    from statsmodels.tsa.api import VAR

    class SVARModel:
        def __init__(self, data):
            self.data = data
            self.var_model = None
            self.svar_results = None
        def fit(self, lags=1, A=None, B=None):
            """Fit SVAR model with short-run (A) and long-run (B) restrictions"""
            self.var_model = VAR(self.data)
            var_results = self.var_model.fit(lags)
            if A is None:
                A = np.eye(len(self.data.columns))
            if B is None:
                B = np.eye(len(self.data.columns))
            self.svar_results = var_results.svar(A=A, B=B)
            return self.svar_results
        def impulse_response(self, periods=20):
            """Calculate impulse response functions"""
            return self.svar_results.irf(periods=periods)
        def forecast_error_variance_decomposition(self, periods=20):
            """Compute forecast error variance decomposition"""
            return self.svar_results.fevd(periods=periods)

## Local Projections for Causal Analysis

Local projections provide a flexible alternative for estimating impulse responses without relying on parametric VAR assumptions.

    import statsmodels.api as sm

    class LocalProjections:
        def __init__(self, data):
            self.data = data
        def estimate_impulse_response(self, dependent_var, shock_var, controls=None, horizons=20):
            """Estimate impulse responses using local projections"""
            responses = []
            confidence_intervals = []
            for h in range(horizons + 1):
                y = self.data[dependent_var].shift(-h)
                X = self.data[[shock_var]]
                if controls is not None:
                    X = pd.concat([X, self.data[controls]], axis=1)
                X = sm.add_constant(X)
                valid_idx = y.notna()
                y = y[valid_idx]
                X = X[valid_idx]
                model = sm.OLS(y, X)
                results = model.fit(cov_type='HAC', cov_kwds={'maxlags': h})
                responses.append(results.params[shock_var])
                confidence_intervals.append(results.conf_int().loc[shock_var])
            return np.array(responses), np.array(confidence_intervals)

## Synthetic Control Method

The synthetic control method constructs a synthetic counterfactual for causal analysis.

    from scipy.optimize import minimize

    class SyntheticControl:
        def __init__(self, data, treatment_unit, control_units, treatment_period, outcome_var):
            self.data = data
            self.treatment_unit = treatment_unit
            self.control_units = control_units
            self.treatment_period = treatment_period
            self.outcome_var = outcome_var
        def construct_synthetic_control(self):
            """Construct synthetic control unit"""
            pre_treatment = self.data[self.data.index < self.treatment_period]
            def objective(weights):
                synthetic = np.sum([
                    w * pre_treatment[self.outcome_var][pre_treatment.unit == u]
                    for w, u in zip(weights, self.control_units)
                ], axis=0)
                treated = pre_treatment[self.outcome_var][
                    pre_treatment.unit == self.treatment_unit
                ]
                return np.mean((treated - synthetic) ** 2)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x}
            ]
            result = minimize(
                objective,
                x0=np.ones(len(self.control_units)) / len(self.control_units),
                constraints=constraints
            )
            return result.x

We need to be cautious about imputing causality. In addition to the math, we need to consider:

- Temporal ordering and dynamics

- Identification assumptions

- Endogeneity concerns

- Structural breaks and regime changes

- Heterogeneous treatment effects

- Policy interventions and external validity

Modern approaches combine traditional econometric methods with recent advances in causal inference, providing robust tools for analyzing economic relationships. Ultimately, the credibility of causal claims depends on the strength of identifying assumptions and data quality.

## Key Takeaways

- Temporal ordering and dynamics
- Identification assumptions
- Endogeneity concerns
- Structural breaks and regime changes
