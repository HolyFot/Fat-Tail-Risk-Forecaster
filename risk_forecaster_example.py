#!/usr/bin/env python3
"""
Risk Forecaster Example Script

This script demonstrates how to use the risk_forecaster.py module for
comprehensive risk analysis and forecasting of cryptocurrency data.

Features demonstrated:
- Data downloading and preprocessing
- Adaptive tail risk analysis
- Consensus risk forecasting
- Risk chart generation
- Backtesting and model validation
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the risk forecasting modules
from risk_forecaster import (
    AdaptiveTailRiskAnalyzer,
    ConsensusRiskForecaster,
    RiskChartGenerator,
    BacktestingEngine,
    DataCleaner,
    ReturnTransformer,
    RiskDataPipeline,
    StressTestRunner
)

class BTCDataDownloader:
    """
    Bitcoin OHLCV data downloader with caching capabilities.
    """
    
    def __init__(self, symbol='BTC/USD', exchange_name='binance', 
                 start_date='2024-01-01', end_date='2025-09-26', cache_dir='./risk_reports/data'):
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        try:
            import ccxt
            if exchange_name.lower() == 'binance':
                self.exchange = ccxt.binance({'enableRateLimit': True})
            elif exchange_name.lower() == 'coinbase':
                self.exchange = ccxt.coinbase({'enableRateLimit': True})
            else:
                self.exchange = getattr(ccxt, exchange_name.lower())({'enableRateLimit': True})
        except ImportError:
            print("Warning: ccxt not installed. Using dummy data.")
            self.exchange = None
        
        self.ohlcv_data = None
        self.prices = None
        self.returns = None
        self.dates = None

    def _get_cache_filename(self, timeframe='15m'):
        """Generate cache filename including timeframe."""
        symbol_clean = self.symbol.replace('/', '_')
        return self.cache_dir / f"{self.exchange_name}_{symbol_clean}_{self.start_date}_{self.end_date}_{timeframe}.parquet"

    def _save_to_cache(self, df, timeframe='15m'):
        """Save DataFrame to cache with timeframe specification."""
        cache_file = self._get_cache_filename(timeframe)
        try:
            df.write_parquet(cache_file)
            print(f"✅ Cached {timeframe} data to {cache_file}")
        except Exception as e:
            print(f"❌ Error saving to cache: {e}")

    def _load_from_cache(self, timeframe='15m'):
        """Load data from cache with timeframe specification."""
        cache_file = self._get_cache_filename(timeframe)
        if not cache_file.exists():
            return None
        
        try:
            df = pl.read_parquet(cache_file)
            if 'index' not in df.columns:
                df = df.with_row_index('index')
            print(f"✅ Loaded cached {timeframe} data from {cache_file}")
            return df
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return None

    def download_data(self, timeframe='15m', force_refresh=False):
        """Download cryptocurrency OHLCV data."""
        # Map unsupported timeframes to supported ones
        timeframe_mapping = {
            '24h': '1d',  # Coinbase uses '1d' instead of '24h'
            '6h': '1h',   # Coinbase doesn't support 6h, use 1h
            '3h': '1h',   # Coinbase doesn't support 3h, use 1h
            '1h': '1h',
            '30m': '30m',
            '15m': '15m',
            '5m': '5m',
            '1m': '1m'
        }
        
        mapped_timeframe = timeframe_mapping.get(timeframe, timeframe)
        
        # Try to load from cache first
        if not force_refresh:
            cached_data = self._load_from_cache(timeframe)
            if cached_data is not None:
                self.ohlcv_data = cached_data
                self._process_ohlcv_data()
                return
        
        if self.exchange is None:
            print("📊 Creating dummy data for demonstration...")
            self._create_dummy_data()
            return
        
        try:
            # Simple data fetching for recent data
            print(f"📊 Downloading {self.symbol} {timeframe} data (using {mapped_timeframe})...")
            
            # Load markets
            self.exchange.load_markets()
            
            # Fetch recent data (last 1000 candles)
            ohlcv_data = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=mapped_timeframe,  # Use mapped timeframe
                limit=1000
            )
            
            if not ohlcv_data:
                raise ValueError(f"No OHLCV data retrieved for {self.symbol}")
            
            # Convert to DataFrame
            df = pl.DataFrame(ohlcv_data, schema=['timestamp', 'open', 'high', 'low', 'close', 'volume'], orient="row")
            df = df.with_columns(
                pl.col('timestamp').cast(pl.Int64).cast(pl.Datetime('ms'))
            ).with_row_index('index')
            
            print(f"✅ Downloaded {df.height:,} candles")
            
            # Save to cache and process
            self._save_to_cache(df, timeframe)
            self.ohlcv_data = df
            self._process_ohlcv_data()
            
        except Exception as e:
            print(f"❌ Data download failed: {e}")
            # Create minimal dummy data for testing
            self._create_dummy_data()

    def _create_dummy_data(self):
        """Create minimal dummy data for testing."""
        print("📊 Creating dummy BTC data for demonstration...")
        n_points = 1000
        base_price = 65000.0  # Current BTC price range
        
        # Generate realistic Bitcoin-like price data with volatility clustering
        np.random.seed(42)  # For reproducible results
        
        # Create volatility clustering similar to crypto markets
        volatility = np.random.gamma(2, 0.01, n_points)  # Time-varying volatility
        returns = np.random.normal(0, 1, n_points) * volatility
        
        # Add some extreme events (tail risk)
        extreme_events = np.random.choice(n_points, size=int(n_points * 0.02), replace=False)
        returns[extreme_events] = np.random.choice([-1, 1], size=len(extreme_events)) * np.random.uniform(0.05, 0.15, len(extreme_events))
        
        # Generate price series
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        # Create OHLC data
        timestamps = [datetime.now() - timedelta(minutes=15*i) for i in reversed(range(n_points))]
        ohlc_data = []
        
        for i in range(n_points):
            close_price = prices[i+1]
            open_price = prices[i]
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
            volume = np.random.uniform(50, 500)
            
            ohlc_data.append({
                'timestamp': timestamps[i],
                'open': float(open_price),
                'high': float(high_price), 
                'low': float(low_price),
                'close': float(close_price),
                'volume': float(volume)
            })
        
        self.ohlcv_data = pl.DataFrame(ohlc_data).with_row_index('index')
        self._process_ohlcv_data()
        print(f"✅ Created dummy dataset with {len(ohlc_data)} candles")

    def _process_ohlcv_data(self):
        """Process OHLCV data with validation."""
        if self.ohlcv_data is None:
            raise ValueError("No OHLCV data available")
        
        # Extract data
        self.prices = self.ohlcv_data.get_column('close').to_numpy()
        self.dates = np.arange(len(self.prices))
        
        # Calculate log returns
        if len(self.prices) > 1:
            price_ratios = self.prices[1:] / self.prices[:-1]
            self.returns = np.log(price_ratios)
        else:
            self.returns = np.array([])

    def get_pandas_df(self):
        """Convert Polars DataFrame to Pandas for analysis."""
        if self.ohlcv_data is None:
            raise ValueError("No data available. Run download_data() first.")
        return self.ohlcv_data.to_pandas()


def example_risk_analysis():
    """
    Main demonstration function showing comprehensive risk analysis workflow.
    """
    print("=" * 80)
    print("🚀 RISK FORECASTER DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Data Collection and Preparation
    print("\n📊 STEP 1: Data Collection and Preparation")
    print("-" * 50)
    
    # Download BTC data
    downloader = BTCDataDownloader(
        symbol='BTC/USD',
        exchange_name='coinbase',
        cache_dir='./risk_reports/data'
    )
    
    downloader.download_data(timeframe='15m', force_refresh=False)
    
    # Get the data
    df = downloader.get_pandas_df()
    returns = downloader.returns
    
    print(f"✅ Data loaded: {len(df)} observations")
    print(f"✅ Returns computed: {len(returns)} periods")
    print(f"✅ Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"✅ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Step 2: Data Cleaning and Preprocessing
    print("\n🧹 STEP 2: Data Cleaning and Preprocessing")
    print("-" * 50)
    
    # Initialize data pipeline
    pipeline = RiskDataPipeline(cache_dir='./risk_reports/data')
    
    # Process the data comprehensively
    processed_data = pipeline.process_comprehensive(
        raw_data=df,
        symbol='BTC-USD',
        price_column='close'
    )
    
    clean_returns = processed_data['returns']
    features = processed_data['features']
    metadata = processed_data['metadata']
    
    print(f"✅ Data cleaned: {len(clean_returns)} clean returns")
    print(f"✅ Features extracted: {len(features)} feature sets")
    print(f"✅ Skewness: {metadata['statistics']['skewness']:.3f}")
    print(f"✅ Kurtosis: {metadata['statistics']['kurtosis']:.3f}")
    print(f"✅ Extreme observations: {metadata['statistics']['extreme_count']}")
    
    # Step 3: Adaptive Tail Risk Analysis
    print("\n📈 STEP 3: Adaptive Tail Risk Analysis")
    print("-" * 50)
    
    # Initialize the adaptive tail risk analyzer
    tail_analyzer = AdaptiveTailRiskAnalyzer(
        use_evt=True,
        baseline_volatility=35.0,
        use_consensus_forecaster=True,
        cache_dir='./risk_reports'
    )
    
    # Fit the model on historical data
    tail_analyzer.fit(clean_returns)
    
    # Get current risk assessment
    recent_returns = clean_returns[-100:]  # Last 100 observations
    risk_assessment = tail_analyzer.get_current_risk_assessment(
        recent_returns=recent_returns,
        confidence_levels=[0.95, 0.99]
    )
    
    print(f"✅ Model fitted on {len(clean_returns)} observations")
    print(f"✅ Current Risk Level: {risk_assessment['risk_classification']}")
    print(f"✅ Tail Score: {risk_assessment['tail_score']:.3f}")
    print(f"✅ VaR (95%): {risk_assessment['var_estimates']['95%']:.4f}")
    print(f"✅ VaR (99%): {risk_assessment['var_estimates']['99%']:.4f}")
    print(f"✅ Expected Shortfall (95%): {risk_assessment['expected_shortfall']['95%']:.4f}")
    
    # Step 4: Consensus Risk Forecasting
    print("\n🔮 STEP 4: Consensus Risk Forecasting")
    print("-" * 50)
    
    # Initialize consensus forecaster
    consensus_forecaster = ConsensusRiskForecaster(
        aggregation_method='trimmed_mean',
        trim_percentage=0.1
    )
    
    # Fit the consensus model
    consensus_forecaster.fit(clean_returns)
    
    # Generate consensus forecast
    consensus_forecast = consensus_forecaster.forecast_consensus(
        horizon=1,
        confidence_levels=[0.95, 0.99],
        returns_for_features=recent_returns
    )
    
    print(f"✅ Consensus model fitted")
    print(f"✅ Forecast confidence: {consensus_forecast['forecast_confidence']:.3f}")
    print(f"✅ Consensus VaR (95%): {consensus_forecast['consensus_var_95']:.4f}")
    print(f"✅ Consensus VaR (99%): {consensus_forecast['consensus_var_99']:.4f}")
    print(f"✅ Consensus ES (95%): {consensus_forecast['consensus_es_95']:.4f}")
    
    # Display risk attribution
    attribution = consensus_forecast['risk_attribution']
    print(f"✅ EVT contribution: {attribution['evt_contribution']:.1%}")
    print(f"✅ SQR contribution: {attribution['sqr_contribution']:.1%}")
    print(f"✅ Regime contribution: {attribution['regime_contribution']:.1%}")
    
    # Step 5: Model Backtesting and Validation
    print("\n🔍 STEP 5: Model Backtesting and Validation")
    print("-" * 50)
    
    # Initialize backtesting engine
    backtest_engine = BacktestingEngine()
    
    # Run comprehensive backtest
    backtest_results = backtest_engine.rolling_backtest(
        model=consensus_forecaster,
        data=clean_returns,
        window_size=500,
        refit_frequency=22,
        forecast_horizon=1,
        confidence_levels=[0.95, 0.99]
    )
    
    print(f"✅ Backtest completed: {backtest_results['total_forecasts']} forecasts")
    print(f"✅ VaR (95%) violation rate: {backtest_results['violation_rates']['95%']:.3f}")
    print(f"✅ VaR (99%) violation rate: {backtest_results['violation_rates']['99%']:.3f}")
    print(f"✅ Average forecast time: {backtest_results['avg_forecast_time_ms']:.1f}ms")
    print(f"✅ Model accuracy score: {backtest_results['overall_score']:.3f}")
    
    # Step 6: Stress Testing
    print("\n💥 STEP 6: Stress Testing")
    print("-" * 50)
    
    # Initialize stress test runner
    stress_tester = StressTestRunner()
    
    # Run comprehensive stress tests
    stress_results = stress_tester.run_comprehensive_stress_tests(consensus_forecaster)
    
    print(f"✅ Stress tests completed: {len(stress_results)} scenarios")
    
    for scenario_name, results in stress_results.items():
        print(f"✅ {scenario_name}:")
        print(f"   - Max violation rate: {results['max_violation_rate_95']:.3f}")
        print(f"   - Stress performance: {results['stress_performance_score']:.3f}")
    
    # Step 7: Risk Chart Generation
    print("\n📊 STEP 7: Risk Chart Generation")
    print("-" * 50)
    
    # Initialize chart generator
    chart_generator = RiskChartGenerator(output_dir='./risk_reports/charts')
    
    # Prepare historical data for charting
    historical_data = {
        'returns': clean_returns,
        'prices': downloader.prices,
        'dates': pd.date_range(start='2024-01-01', periods=len(clean_returns), freq='15min'),
        'metadata': metadata
    }
    
    # Generate comprehensive risk charts
    chart_files = chart_generator.generate_comprehensive_risk_charts(
        forecaster=consensus_forecaster,
        historical_data=historical_data,
        symbol='BTC/USD'
    )
    
    print(f"✅ Charts generated: {len(chart_files)} files")
    for chart_type, filepath in chart_files.items():
        print(f"   - {chart_type}: {filepath}")
    
    # Step 8: Generate Risk Reports
    print("\n📋 STEP 8: Generate Risk Reports")
    print("-" * 50)
    
    # Generate risk analysis report
    risk_report = tail_analyzer.generate_risk_report(
        recent_returns=recent_returns,
        symbol='BTC/USD'
    )
    
    # Save the risk report
    report_path = Path('./risk_reports/btc_risk_analysis_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(risk_report)
    
    print(f"✅ Risk report saved: {report_path}")
    
    # Generate backtest report
    backtest_report = backtest_engine.generate_backtest_report(backtest_results)
    
    backtest_report_path = Path('./risk_reports/btc_backtest_report.txt')
    with open(backtest_report_path, 'w') as f:
        f.write(backtest_report)
    
    print(f"✅ Backtest report saved: {backtest_report_path}")
    
    # Step 9: Summary and Next Steps
    print("\n🎯 STEP 9: Analysis Summary")
    print("-" * 50)
    
    print("✅ ANALYSIS COMPLETE!")
    print(f"\n📊 Key Risk Metrics:")
    print(f"   • Current Risk Level: {risk_assessment['risk_classification']}")
    print(f"   • Tail Score: {risk_assessment['tail_score']:.3f}")
    print(f"   • 1-Day VaR (95%): {consensus_forecast['consensus_var_95']:.4f} ({consensus_forecast['consensus_var_95']*100:.2f}%)")
    print(f"   • 1-Day VaR (99%): {consensus_forecast['consensus_var_99']:.4f} ({consensus_forecast['consensus_var_99']*100:.2f}%)")
    print(f"   • Model Confidence: {consensus_forecast['forecast_confidence']:.3f}")
    
    print(f"\n📈 Portfolio Risk Insights:")
    dollar_value = 100000  # $100k portfolio example
    var_95_dollar = dollar_value * abs(consensus_forecast['consensus_var_95'])
    var_99_dollar = dollar_value * abs(consensus_forecast['consensus_var_99'])
    
    print(f"   • For a $100,000 BTC position:")
    print(f"   • Max daily loss (95% confidence): ${var_95_dollar:,.2f}")
    print(f"   • Max daily loss (99% confidence): ${var_99_dollar:,.2f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • Risk charts: ./risk_reports/charts/")
    print(f"   • Risk analysis report: {report_path}")
    print(f"   • Backtest report: {backtest_report_path}")
    print(f"   • Cached data: ./risk_reports/data/")
    
    print(f"\n🔄 Next Steps:")
    print(f"   • Review generated charts for visual risk analysis")
    print(f"   • Implement real-time monitoring using the fitted models")
    print(f"   • Set up automated risk reporting")
    print(f"   • Integrate with trading systems for position sizing")
    
    return {
        'risk_assessment': risk_assessment,
        'consensus_forecast': consensus_forecast,
        'backtest_results': backtest_results,
        'stress_results': stress_results,
        'chart_files': chart_files,
        'models': {
            'tail_analyzer': tail_analyzer,
            'consensus_forecaster': consensus_forecaster
        }
    }


def example_live_monitoring():
    """
    Demonstrate how to use the models for live risk monitoring.
    """
    print("\n" + "=" * 80)
    print("🔴 LIVE MONITORING DEMONSTRATION")
    print("=" * 80)
    
    # This would typically be called in a loop for real-time monitoring
    downloader = BTCDataDownloader()
    downloader.download_data(timeframe='15m')
    
    # Initialize models (would be loaded from saved state in production)
    consensus_forecaster = ConsensusRiskForecaster()
    consensus_forecaster.fit(downloader.returns)
    
    # Get latest risk assessment
    latest_returns = downloader.returns[-50:]  # Last 50 periods
    
    live_forecast = consensus_forecaster.forecast_consensus(
        horizon=1,
        confidence_levels=[0.95, 0.99],
        returns_for_features=latest_returns
    )
    
    print(f"🔴 LIVE RISK ALERT")
    print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   • Current BTC Price: ${downloader.prices[-1]:,.2f}")
    print(f"   • Risk Level: {live_forecast.get('risk_level', 'MODERATE')}")
    print(f"   • 1-Day VaR (95%): {live_forecast['consensus_var_95']*100:.2f}%")
    print(f"   • 1-Day VaR (99%): {live_forecast['consensus_var_99']*100:.2f}%")
    print(f"   • Forecast Confidence: {live_forecast['forecast_confidence']:.3f}")
    
    # Risk-based position sizing example
    account_balance = 100000  # $100k account
    risk_tolerance = 0.02  # 2% daily VaR
    
    max_position_size = (account_balance * risk_tolerance) / abs(live_forecast['consensus_var_95'])
    
    print(f"\n💰 POSITION SIZING RECOMMENDATION:")
    print(f"   • Account Balance: ${account_balance:,}")
    print(f"   • Risk Tolerance: {risk_tolerance*100:.1f}% daily VaR")
    print(f"   • Recommended Max Position: ${max_position_size:,.2f}")
    print(f"   • Max BTC Units: {max_position_size/downloader.prices[-1]:.4f} BTC")


if __name__ == "__main__":
    """
    Run the complete risk forecasting demonstration.
    """
    try:
        # Run main demonstration
        results = example_risk_analysis()
        
        # Show live monitoring example
        example_live_monitoring()
        
        print("\n" + "=" * 80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nCheck the './risk_reports/' directory for all generated files.")
        print("\nTo use in production:")
        print("1. Replace dummy data with real exchange data")
        print("2. Set up automated data collection")
        print("3. Implement real-time alerting")
        print("4. Integrate with trading systems")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

