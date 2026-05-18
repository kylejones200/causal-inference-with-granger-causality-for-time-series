# Repository

Companion code for a Medium article.

## Business context

Correlation is easy to find in time series data. Two variables that both trend upward will correlate. Two variables driven by the same seasonal cycle will correlate. None of that tells you anything about causation. Causal inference methods try to answer a harder question: if you intervened on variable X, what would happen to variable Y?

For time series, the toolbox includes Granger causality, structural VAR models, local projections, and the synthetic control method. Each makes different assumptions and answers a slightly different question. None of them guarantee true causation — that requires a credible identification argument, not just a statistical test.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).