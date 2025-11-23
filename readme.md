# Risk Forecasting & Backtesting System

Fat-Tail Risk forecasting system combining TVP-EVT & SQR models for tail risk prediction with seasonal quantile regression.
This system integrates Time-Varying Parameter Extreme Value Theory (TVP-EVT) and Seasonal Quantile Regression (SQR).
Both models train off the OHLCV data you give it.
Note: VaR 95 is very accurate whil VaR 99 can be inaccurate.

## System Overview

- **Comprehensive backtesting** with statistical validation (ACF/PACF
- **Live forecasting**
- **Optimized with Cython & Polars**
- **Multi-timeframe analysis** from 1-minute to 24-hour horizons
- **Value at Risk (VaR)**: Multiple confidence levels with seasonal adjustment

## Architecture

### Core Model Components

#### 1. TVP-EVT Model (Tail Dynamics Engine)
- **Framework**: Time-varying extreme value theory with state-space evolution
- **Key Features**: Dynamic threshold selection, Bayesian parameter tracking, tail risk forecasting
- **Output**: Posterior distributions, tail risk metrics, credible intervals

**Forecasting VaR / ES**:
- VaR_t+1(p) = u_t + (beta_t / xi_t) * [ (N_t / (n_t * (1 - p)))^xi_t - 1 ]
- ES_t+1(p) = (VaR_t+1(p) / (1 - xi_t)) + (beta_t - xi_t * u_t) / (1 - xi_t)

- VaR_t+1(p) gives the cutoff loss level such that only (1 – p) × 100% of losses exceed it.
- ES_t+1(p) gives the expected loss if that extreme event occurs, using the same evolving tail parameters.
- Evolves producing time-varying forecasts of tail risk.

#### 2. Seasonal Quantile Regression (Conditional Risk Layer)
- **Framework**: Multi-harmonic VaR modeling with seasonal decomposition
- **Features**: Quantile optimization, seasonal pattern detection
- **Output**: Conditional VaR forecasts, seasonal risk decomposition, feature importance

- Regress conditional VaR/ES on seasonal factors and EVT posteriors:
-   VaRt+1,0.01=α+β1ξt+β2βt+γ1sin(2πt/T)+γ2cos(2πt/T)
-   Captures cyclical risk elevation (e.g., BTC weekend effect, Q4 rally).

### Model Consensus
The `ConsensusRiskForecaster` combines both model outputs using weighting based on:
- Recent forecast accuracy (MSE, violation rates)
- Model confidence/uncertainty metrics

## Performance Metrics

### Success Criteria
- **Statistical Performance**: Violation rate within 95% CI of target
- **Economic Performance**: Lower capital requirements vs. historical simulation
- **Operational Performance**: Prediction latency < 100ms, System uptime > 99.9%


## Integration

### Data Sources
The system can integrate with various data providers:
- **Yahoo Finance** (`yfinance`)
- **Cryptocurrency exchanges** (`ccxt`)
- **Custom CSV/Parquet files**
- **Real-time market data streams**

## Documentation

### Key Classes
- `AdaptiveTailRiskAnalyzer`: Main risk analysis interface
- `ConsensusRiskForecaster`: Multi-model ensemble forecasting
- `BacktestingEngine`: Comprehensive model validation
- `LiveRiskForecaster`: Real-time forecasting system
- `OptimizedModelFactory`: High-performance model creation

### Mathematical Framework
The system implements state-of-the-art financial risk models:
- **TVP-EVT**: Time-varying extreme value theory with Bayesian updating
- **Seasonal QR**: Multi-harmonic quantile regression with seasonal decomposition


## Installation

### Dependencies

```bash
pip install numpy pandas polars scipy matplotlib seaborn
pip install scikit-learn statsmodels arch
pip install cython numba
pip install cython

# Optional (used in live test)
pip install flask
pip install yfinance ccxt
```

Compile the Cython modules:

```bash
python build_risk.py
```

## Quick Start

### Basic Risk Analysis

```python
from risk_forecaster import AdaptiveTailRiskAnalyzer
import polars as pl

# Initialize the analyzer
analyzer = AdaptiveTailRiskAnalyzer(
    use_evt=True,
    use_consensus_forecaster=True,
    cache_dir='./risk_cache'
)

# Load your data (returns as Polars DataFrame)
returns_df = pl.read_csv('your_returns.csv')

# Calculate adaptive tail risk
results = analyzer.calculate_adaptive_tail_risk(
    returns=returns_df,
    confidence_levels=[0.95, 0.99]
)

print(results)
```

### Live Forecasting System

```python
from risk_forecaster_live_test import LiveRiskForecaster

# Initialize live forecaster
forecaster = LiveRiskForecaster(
    symbol='BTC/USD',
    exchange_name='coinbase'
)

# Initialize system with historical data
forecaster.initialize_system()

# Start live forecasting
forecaster.start_live_forecasting()

# Get real-time dashboard data
dashboard_data = forecaster.get_dashboard_data()
print(dashboard_data['performance_summary'])
```

### Consensus Training

```python
from risk_forecaster import ConsensusRiskForecaster, BacktestingEngine

# Initialize consensus model
consensus = ConsensusRiskForecaster(
    tvp_evt_params={'threshold_method': 'hill'},
    sqr_params={'quantiles': [0.01, 0.025, 0.05, 0.1]},
)

# Fit on historical data
consensus.fit(returns_data)

# Generate forecasts
forecast = consensus.forecast_consensus(horizon=1)
print(f"VaR Forecast: {forecast['var']}")
```

## Key Features

### Advanced Risk Metrics
- **Value at Risk (VaR)**: Multiple confidence levels with seasonal adjustment
- **Expected Shortfall**: Coherent risk measure beyond VaR
- **Tail Risk**: Extreme value theory-based tail risk quantification

### Backtesting Engine
```python
from risk_forecaster import BacktestingEngine

engine = BacktestingEngine()

# Comprehensive backtesting
results = engine.backtest_comprehensive(
    model=consensus,
    data=historical_data,
    params={
        'window_size': 252,
        'refit_freq': 22,
        'horizon': 1,
        'var_level': 0.05
    }
)

# Statistical validation
print(f"Kupiec Test p-value: {results['coverage_tests']['kupiec']['p_value']}")
print(f"Violation Rate: {results['metrics']['violation_rate']:.2%}")
```

### Multi-Timeframe Analysis
The system supports forecasting across multiple timeframes:
- **High-frequency**: 1m, 5m, 15m, 30m
- **Intraday**: 1h, 3h, 6h
- **Daily**: 24h


## Configuration

### Model Parameters

#### TVP-EVT Configuration
```python
tvp_evt_params = {
    'innovation_variance_xi': 0.01,
    'innovation_variance_beta': 0.01,
    'persistence_beta': 0.95,
    'threshold_method': 'hill',
    'n_particles': 1000
}
```

#### Seasonal Quantile Regression
```python
sqr_params = {
    'quantiles': [0.01, 0.025, 0.05, 0.1],
    'n_harmonics': 5,
    'regularization': 'l1',
    'seasonal_periods': [252, 22, 5]  # Annual, monthly, weekly
}
```


## Live Dashboard

Launch the web-based dashboard for real-time monitoring:

```python
from risk_forecaster_live_test import TailRiskFlaskServer, LiveRiskForecaster

# Initialize components
forecaster = LiveRiskForecaster()
server = TailRiskFlaskServer(forecaster)

# Run dashboard
server.run(host='127.0.0.1', port=5002, debug=False)
```

Access dashboard at: `http://localhost:5002`

### API Endpoints
- `/api/live_status` - Live forecasting status
- `/api/accuracy_stats` - Detailed accuracy statistics
- `/api/performance_summary` - Performance summary
- `/api/recent_forecasts/<timeframe>` - Recent forecasts by timeframe
- `/api/system_control/<action>` - Start/stop forecasting

## Testing & Validation

### Statistical Tests
The system includes comprehensive statistical validation:
- **ACF/PACF**: Auto Correlation Function/Partial Auto Correlation Function validation
- **Kupiec Test**: Unconditional coverage test
- **Christoffersen Test**: Conditional coverage test
- **Dynamic Quantile Test**: Time-varying coverage validation

### Stress Testing
```python
# Stress testing scenarios
stress_scenarios = [
    'historical_crises',  # 2008, 2020, etc.
    'synthetic_jumps',    # Generated tail events
    'seasonal_shocks'     # Holiday crashes
]

stress_results = engine.stress_test(model, scenarios=stress_scenarios)
```

## Project Structure

```
risk-forecasting-system/
├── risk_forecaster.py           # Main analyzer and models
├── risk_forecaster_live_test.py # Live forecasting system
├── risk_forecaster_example.py   # Example on how to use risk_forecaster.py
├── risk_engine.pyx              # Cython optimization modules
├── README.md                    # This file
├── risk_charts/                 # Stores charts generated
└── data/                        # Sample OHLCV Data
```
