import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import warnings
import os
import glob
from itertools import combinations
import seaborn as sns
import time

warnings.filterwarnings('ignore')

# Font settings
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Set random seed
np.random.seed(42)

# Data paths
desktop_path = r"C:\Users\13616\Desktop"
data_folder = r"C:\Users\13616\Desktop\raw_data"


class StockForecastResearchSystem:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.all_stocks = []
        self.h1_results = []
        self.h2_results = []
        self.h3_results = []
        self.vftse_data = None
        self.figures_saved = []
        self.error_log = []
        self.feature_importance_data = {}
        self.prediction_examples = {}

    def load_stock_data(self, file_path):
        """Load stock price data from CSV file"""
        print(f"Reading stock data file: {os.path.basename(file_path)}")
        print("=" * 80)

        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            self.original_data = self.data.copy()

            print(f"Successfully read data")
            print(f"Original data size: {self.data.shape}")

            if 'Date' in self.data.columns or 'date' in self.data.columns:
                date_col = 'Date' if 'Date' in self.data.columns else 'date'
                self.data[date_col] = pd.to_datetime(self.data[date_col])
                self.data = self.data.sort_values(date_col).reset_index(drop=True)
                print(f"Date column processed: {date_col}")

            numeric_cols = []
            for col in self.data.columns:
                if col.lower() not in ['date', 'unnamed: 0', 'index']:
                    try:
                        temp_series = pd.to_numeric(self.data[col], errors='coerce')
                        if temp_series.notna().sum() > 100:
                            numeric_cols.append(col)
                    except:
                        continue

            self.all_stocks = numeric_cols
            print(f"Found stocks: {len(self.all_stocks)}")
            print(f"Data time span: {len(self.data)} time points")

            if len(self.all_stocks) == 0:
                raise Exception("No valid stock data found")

            for col in self.all_stocks:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

            self.data = self.data.dropna(how='all', subset=self.all_stocks)
            print(f"Cleaned data: {len(self.data)} rows")
            print(f"Final stocks: {len(self.all_stocks)}")

            self.load_vftse_data()
            return True

        except Exception as e:
            print(f"Data loading failed: {e}")
            self.error_log.append(f"Data loading failed: {e}")
            return False

    def load_vftse_data(self):
        """Load VFTSE volatility index data"""
        try:
            vftse_file = os.path.join(data_folder, "FTSE100_Volatility_2014_2024.csv")
            if os.path.exists(vftse_file):
                vftse_df = pd.read_csv(vftse_file)
                if len(vftse_df.columns) >= 2:
                    self.vftse_data = vftse_df.iloc[:, 1].values
                    print(f"VFTSE sentiment indicator loaded: {len(self.vftse_data)} data points")
                else:
                    self.vftse_data = None
            else:
                self.vftse_data = None
                print("VFTSE file not found, will simulate sentiment indicator")
        except Exception as e:
            self.vftse_data = None
            print(f"VFTSE loading failed: {e}")

    def create_vftse_sentiment(self, stock_prices):
        """Create simulated VFTSE sentiment indicator"""
        if self.vftse_data is not None and len(self.vftse_data) >= len(stock_prices):
            return self.vftse_data[:len(stock_prices)]
        else:
            returns = pd.Series(stock_prices).pct_change().fillna(0)
            rolling_vol = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)
            vftse_raw = rolling_vol * 100
            return vftse_raw.values

    def create_market_regime(self, vftse_values):
        """Create market regime classification"""
        try:
            q75 = np.quantile(vftse_values, 0.75)
            q25 = np.quantile(vftse_values, 0.25)

            regime = []
            for s_t in vftse_values:
                if s_t >= q75:
                    regime.append('HighVol')
                elif s_t <= q25:
                    regime.append('LowVol')
                else:
                    regime.append('Mid')

            return np.array(regime), {'q75': q75, 'q25': q25}

        except Exception as e:
            print(f"Market regime stratification failed: {e}")
            return np.array(['Mid'] * len(vftse_values)), {'q75': 0, 'q25': 0}

    def create_target_variable(self, prices, H=5):
        """Create target variable: y_t(H) = ln P_{t+H} - ln P_t"""
        try:
            log_prices = np.log(prices + 1e-8)
            target = []

            for i in range(len(log_prices) - H):
                y_h = log_prices[i + H] - log_prices[i]
                target.append(y_h)

            return np.array(target)
        except Exception as e:
            print(f"Target variable creation failed: {e}")
            return np.array([])

    def create_feature_set(self, stock_prices, feature_type="full", include_vftse=True):
        """Create feature set for model training"""
        try:
            prices = pd.Series(stock_prices)
            features = pd.DataFrame()

            # Basic price features
            if feature_type in ["price_only", "price_tech", "full"]:
                features['Price'] = prices
                features['Log_Price'] = np.log(prices + 1e-8)
                returns = prices.pct_change().fillna(0)
                features['Return'] = returns
                features['Return_Lag1'] = returns.shift(1).fillna(0)
                features['Return_Lag2'] = returns.shift(2).fillna(0)

            # Technical indicators
            if feature_type in ["price_tech", "full"]:
                features['SMA_15'] = prices.rolling(window=15, min_periods=1).mean()
                features['SMA_45'] = prices.rolling(window=45, min_periods=1).mean()

                delta = prices.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                features['RSI'] = 100 - (100 / (1 + rs))

                ema_12 = prices.ewm(span=12, min_periods=1).mean()
                ema_26 = prices.ewm(span=26, min_periods=1).mean()
                features['MACD'] = ema_12 - ema_26
                features['MACD_Signal'] = features['MACD'].ewm(span=9, min_periods=1).mean()

                bb_middle = prices.rolling(window=20, min_periods=1).mean()
                bb_std = prices.rolling(window=20, min_periods=1).std()
                features['BB_Upper'] = bb_middle + (bb_std * 2)
                features['BB_Lower'] = bb_middle - (bb_std * 2)
                features['BB_Position'] = (prices - features['BB_Lower']) / (
                            features['BB_Upper'] - features['BB_Lower'] + 1e-8)

            # VFTSE sentiment indicator
            if feature_type == "full" and include_vftse:
                vftse = self.create_vftse_sentiment(stock_prices)
                if len(vftse) == len(features):
                    features['VFTSE'] = vftse
                    vftse_mean = np.mean(vftse)
                    vftse_std = np.std(vftse)
                    features['VFTSE_zscore'] = (vftse - vftse_mean) / (vftse_std + 1e-8)

            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)

            return features

        except Exception as e:
            print(f"Feature creation failed: {e}")
            return pd.DataFrame()

    def train_models_with_importance(self, X_train, y_train, X_test, model_types=None):
        """Train models and record feature importance"""
        if model_types is None:
            model_types = ['SVR', 'RF', 'MLP']

        models = {}
        predictions = {}

        try:
            if 'SVR' in model_types:
                models['SVR'] = SVR(C=10, epsilon=0.1, kernel='rbf')
            if 'RF' in model_types:
                models['RF'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            if 'MLP' in model_types:
                models['MLP'] = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    predictions[name] = y_pred

                    if name == 'RF' and hasattr(model, 'feature_importances_'):
                        self.feature_importance_data[name] = model.feature_importances_

                except Exception as e:
                    print(f"Model {name} training failed: {e}")
                    self.error_log.append(f"Model {name} training failed: {e}")
                    continue

        except Exception as e:
            print(f"Model training batch failed: {e}")

        return predictions

    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics for model performance"""
        try:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

            return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Direction_Accuracy': direction_accuracy}
        except Exception as e:
            print(f"Metric calculation failed: {e}")
            return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Direction_Accuracy': np.nan}

    def conduct_h1_hypothesis(self, sample_stocks=50):
        """H1: Model comparison + market regime stratification"""
        print("\nH1 Hypothesis: Model Comparison Study (SVR vs RF vs MLP) + Market Regime Stratification")
        print("=" * 60)

        sample_stocks_list = self.all_stocks[:sample_stocks] if len(
            self.all_stocks) > sample_stocks else self.all_stocks
        h1_results = []

        for i, stock in enumerate(sample_stocks_list):
            if i % 10 == 0:
                print(f"Progress: {i + 1}/{len(sample_stocks_list)} stocks")

            try:
                prices = self.data[stock].dropna().values
                if len(prices) < 200:
                    continue

                features = self.create_feature_set(prices, feature_type="full")
                target = self.create_target_variable(prices, H=5)

                if features.empty or len(target) == 0:
                    continue

                min_len = min(len(features), len(target))
                features = features.iloc[:min_len]
                target = target[:min_len]

                if len(features) < 100:
                    continue

                if 'VFTSE_zscore' in features.columns:
                    market_regime, regime_stats = self.create_market_regime(features['VFTSE_zscore'].values)
                else:
                    market_regime = np.array(['Mid'] * len(features))
                    regime_stats = {'q75': 0, 'q25': 0}

                split_point = int(len(features) * 0.8)
                X_train = features.iloc[:split_point]
                X_test = features.iloc[split_point:]
                y_train = target[:split_point]
                y_test = target[split_point:]
                regime_test = market_regime[split_point:]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                predictions = self.train_models_with_importance(X_train_scaled, y_train, X_test_scaled)

                if i == 0 and len(predictions) > 0:
                    best_model = max(predictions.keys(),
                                     key=lambda x: self.calculate_metrics(y_test, predictions[x])['R2'])
                    self.prediction_examples['y_true'] = y_test[:50]
                    self.prediction_examples['y_pred'] = predictions[best_model][:50]
                    self.prediction_examples['model_name'] = best_model
                    self.prediction_examples['stock_name'] = stock

                for model_name, y_pred in predictions.items():
                    metrics = self.calculate_metrics(y_test, y_pred)
                    h1_results.append({
                        'Stock': stock,
                        'Model': model_name,
                        'Hypothesis': 'H1',
                        'Regime': 'Overall',
                        **metrics
                    })

                    for regime_type in ['HighVol', 'LowVol', 'Mid']:
                        regime_mask = regime_test == regime_type
                        if np.sum(regime_mask) > 5:
                            try:
                                y_test_regime = y_test[regime_mask]
                                y_pred_regime = y_pred[regime_mask]
                                metrics_regime = self.calculate_metrics(y_test_regime, y_pred_regime)
                                h1_results.append({
                                    'Stock': stock,
                                    'Model': model_name,
                                    'Hypothesis': 'H1',
                                    'Regime': regime_type,
                                    **metrics_regime
                                })
                            except Exception as e:
                                continue

            except Exception as e:
                self.error_log.append(f"H1-{stock}: {e}")
                continue

        self.h1_results = h1_results
        print(f"H1 Hypothesis completed: {len(h1_results)} experiments (including stratification)")
        return len(h1_results) > 0

    def conduct_h2_hypothesis(self, sample_stocks=30):
        """H2: Feature engineering effectiveness"""
        print("\nH2 Hypothesis: Feature Engineering Study (Price vs Price+Technical Indicators)")
        print("=" * 60)

        sample_stocks_list = self.all_stocks[:sample_stocks] if len(
            self.all_stocks) > sample_stocks else self.all_stocks
        h2_results = []
        feature_sets = [
            ("Price_Only", "price_only"),
            ("Price_Tech", "price_tech"),
            ("Full_Features", "full")
        ]

        for i, stock in enumerate(sample_stocks_list):
            if i % 10 == 0:
                print(f"Progress: {i + 1}/{len(sample_stocks_list)} stocks")

            try:
                prices = self.data[stock].dropna().values
                if len(prices) < 200:
                    continue

                target = self.create_target_variable(prices, H=5)
                if len(target) == 0:
                    continue

                for feature_name, feature_type in feature_sets:
                    try:
                        features = self.create_feature_set(prices, feature_type=feature_type)

                        if features.empty:
                            continue

                        min_len = min(len(features), len(target))
                        features = features.iloc[:min_len]
                        target_aligned = target[:min_len]

                        if len(features) < 100:
                            continue

                        split_point = int(len(features) * 0.8)
                        X_train = features.iloc[:split_point]
                        X_test = features.iloc[split_point:]
                        y_train = target_aligned[:split_point]
                        y_test = target_aligned[split_point:]

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)

                        metrics = self.calculate_metrics(y_test, y_pred)
                        h2_results.append({
                            'Stock': stock,
                            'FeatureSet': feature_name,
                            'FeatureCount': len(features.columns),
                            'Hypothesis': 'H2',
                            **metrics
                        })

                    except Exception as e:
                        self.error_log.append(f"H2-{stock}-{feature_name}: {e}")
                        continue

            except Exception as e:
                self.error_log.append(f"H2-{stock}: {e}")
                continue

        self.h2_results = h2_results
        print(f"H2 Hypothesis completed: {len(h2_results)} experiments")
        return len(h2_results) > 0

    def conduct_h3_hypothesis(self, sample_stocks=30):
        """H3: Forecast horizon + market regime + multi-model comparison"""
        print("\nH3 Hypothesis: Forecast Horizon Study + Market Regime Stratification + Multi-Model Comparison")
        print("=" * 60)

        sample_stocks_list = self.all_stocks[:sample_stocks] if len(
            self.all_stocks) > sample_stocks else self.all_stocks
        h3_results = []
        horizons = [5, 10, 30, 60]
        model_types = ['SVR', 'RF', 'MLP']

        for i, stock in enumerate(sample_stocks_list):
            if i % 10 == 0:
                print(f"H3 Progress: {i + 1}/{len(sample_stocks_list)} stocks")

            try:
                prices = self.data[stock].dropna().values
                if len(prices) < 300:
                    continue

                for H in horizons:
                    horizon_type = "short" if H <= 10 else "medium" if H <= 30 else "long"

                    try:
                        features = self.create_feature_set(prices, feature_type="full")
                        target = self.create_target_variable(prices, H=H)

                        if features.empty or len(target) == 0:
                            continue

                        min_len = min(len(features), len(target))
                        features = features.iloc[:min_len]
                        target_aligned = target[:min_len]

                        if len(features) < 100:
                            continue

                        if 'VFTSE_zscore' in features.columns:
                            market_regime, _ = self.create_market_regime(features['VFTSE_zscore'].values)
                        else:
                            market_regime = np.array(['Mid'] * len(features))

                        split_point = int(len(features) * 0.8)
                        X_train = features.iloc[:split_point]
                        X_test = features.iloc[split_point:]
                        y_train = target_aligned[:split_point]
                        y_test = target_aligned[split_point:]
                        regime_test = market_regime[split_point:]

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        predictions = self.train_models_with_importance(X_train_scaled, y_train, X_test_scaled,
                                                                        model_types)

                        for model_name, y_pred in predictions.items():
                            metrics = self.calculate_metrics(y_test, y_pred)
                            h3_results.append({
                                'Stock': stock,
                                'Model': model_name,
                                'Horizon_H': H,
                                'Horizon_Type': horizon_type,
                                'Hypothesis': 'H3',
                                'Regime': 'Overall',
                                **metrics
                            })

                            for regime_type in ['HighVol', 'LowVol', 'Mid']:
                                regime_mask = regime_test == regime_type
                                if np.sum(regime_mask) > 5:
                                    try:
                                        y_test_regime = y_test[regime_mask]
                                        y_pred_regime = y_pred[regime_mask]
                                        metrics_regime = self.calculate_metrics(y_test_regime, y_pred_regime)
                                        h3_results.append({
                                            'Stock': stock,
                                            'Model': model_name,
                                            'Horizon_H': H,
                                            'Horizon_Type': horizon_type,
                                            'Hypothesis': 'H3',
                                            'Regime': regime_type,
                                            **metrics_regime
                                        })
                                    except Exception as e:
                                        continue

                    except Exception as e:
                        self.error_log.append(f"H3-{stock}-H{H}: {e}")
                        continue

            except Exception as e:
                self.error_log.append(f"H3-{stock}: {e}")
                continue

        self.h3_results = h3_results
        print(f"H3 Hypothesis completed: {len(h3_results)} experiments (including stratification)")
        return len(h3_results) > 0

    def create_plot1_h1h2_results(self):
        """Create visualization for H1 and H2 results"""
        if not self.h1_results and not self.h2_results:
            print("Skipping Chart 1: No H1 and H2 data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('H1-H2 Hypothesis Research Results', fontsize=16, fontweight='bold')

        if self.h1_results:
            h1_df = pd.DataFrame(self.h1_results)
            h1_overall = h1_df[h1_df['Regime'] == 'Overall']
            if not h1_overall.empty:
                model_performance = h1_overall.groupby('Model')['R2'].agg(['mean', 'std', 'count'])

                bars = axes[0, 0].bar(model_performance.index, model_performance['mean'],
                                      yerr=model_performance['std'], capsize=5, alpha=0.7,
                                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[0, 0].set_title('H1: Model R2 Performance Comparison', fontweight='bold')
                axes[0, 0].set_ylabel('R2 Score')
                axes[0, 0].grid(True, alpha=0.3)

                for bar, mean_val, count in zip(bars, model_performance['mean'], model_performance['count']):
                    axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                    f'{mean_val:.3f}\n(n={count})', ha='center', va='bottom', fontsize=10)
            else:
                axes[0, 0].text(0.5, 0.5, 'No H1 Data Available', ha='center', va='center',
                                transform=axes[0, 0].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, 'No H1 Data Available', ha='center', va='center', transform=axes[0, 0].transAxes)

        if self.h1_results:
            h1_df = pd.DataFrame(self.h1_results)
            h1_overall = h1_df[h1_df['Regime'] == 'Overall']
            if not h1_overall.empty:
                direction_perf = h1_overall.groupby('Model')['Direction_Accuracy'].mean()
                bars = axes[0, 1].bar(direction_perf.index, direction_perf.values, alpha=0.7,
                                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                axes[0, 1].set_title('H1: Direction Prediction Accuracy', fontweight='bold')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Level')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                for bar, acc in zip(bars, direction_perf.values):
                    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                axes[0, 1].text(0.5, 0.5, 'No H1 Direction Data', ha='center', va='center',
                                transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'No H1 Direction Data', ha='center', va='center', transform=axes[0, 1].transAxes)

        if self.h2_results:
            h2_df = pd.DataFrame(self.h2_results)
            feature_performance = h2_df.groupby('FeatureSet')['R2'].agg(['mean', 'std'])

            bars = axes[1, 0].bar(feature_performance.index, feature_performance['mean'],
                                  yerr=feature_performance['std'], capsize=5, alpha=0.7,
                                  color=['#FFA07A', '#98FB98', '#87CEEB'])
            axes[1, 0].set_title('H2: Feature Set R2 Performance Comparison', fontweight='bold')
            axes[1, 0].set_ylabel('R2 Score')
            axes[1, 0].tick_params(axis='x', rotation=15)
            axes[1, 0].grid(True, alpha=0.3)

            for bar, mean_val in zip(bars, feature_performance['mean']):
                axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            axes[1, 0].text(0.5, 0.5, 'No H2 Data Available', ha='center', va='center', transform=axes[1, 0].transAxes)

        if self.h2_results:
            h2_df = pd.DataFrame(self.h2_results)
            feature_count_perf = h2_df.groupby('FeatureCount')['R2'].mean().reset_index()
            if not feature_count_perf.empty:
                axes[1, 1].scatter(feature_count_perf['FeatureCount'], feature_count_perf['R2'],
                                   s=100, alpha=0.7, color='purple')
                axes[1, 1].plot(feature_count_perf['FeatureCount'], feature_count_perf['R2'],
                                color='purple', alpha=0.5)
                axes[1, 1].set_title('H2: Feature Count vs Prediction Performance', fontweight='bold')
                axes[1, 1].set_xlabel('Number of Features')
                axes[1, 1].set_ylabel('R2 Score')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient Feature Data', ha='center', va='center',
                                transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No H2 Feature Data', ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        filename1 = f'{desktop_path}/01_H1H2_hypothesis_results.png'
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close()
        self.figures_saved.append(filename1)

    def create_plot2_h3_results(self):
        """Create visualization for H3 results"""
        if not self.h3_results:
            print("Skipping Chart 2: No H3 data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('H3 Hypothesis Research Results', fontsize=16, fontweight='bold')

        h3_df = pd.DataFrame(self.h3_results)
        h3_overall = h3_df[h3_df['Regime'] == 'Overall']

        if h3_overall.empty:
            for i, ax in enumerate(axes.flat):
                ax.text(0.5, 0.5, 'No H3 Data Available', ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            filename2 = f'{desktop_path}/02_H3_hypothesis_results.png'
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            plt.close()
            self.figures_saved.append(filename2)
            return

        if 'Model' in h3_overall.columns:
            pivot_data = h3_overall.pivot_table(values='R2', index='Model', columns='Horizon_H', aggfunc='mean')
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0, 0],
                            cbar_kws={'label': 'R2 Score'})
                axes[0, 0].set_title('Performance Heatmap: Model × Forecast Horizon', fontweight='bold')
                axes[0, 0].set_xlabel('Forecast Horizon H (Days)')
                axes[0, 0].set_ylabel('Model Type')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Pivot Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Model Column', ha='center', va='center', transform=axes[0, 0].transAxes)

        if 'Model' in h3_overall.columns:
            for model in h3_overall['Model'].unique():
                model_data = h3_overall[h3_overall['Model'] == model]
                horizon_perf = model_data.groupby('Horizon_H')['R2'].mean()
                axes[0, 1].plot(horizon_perf.index, horizon_perf.values, 'o-', label=model, linewidth=2, markersize=6)

            axes[0, 1].set_title('Model Performance Trends by Forecast Horizon', fontweight='bold')
            axes[0, 1].set_xlabel('Forecast Horizon H (Days)')
            axes[0, 1].set_ylabel('R2 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Model Trend Data', ha='center', va='center', transform=axes[0, 1].transAxes)

        if 'Model' in h3_overall.columns:
            models_data = []
            labels = []
            for model in h3_overall['Model'].unique():
                model_r2 = h3_overall[h3_overall['Model'] == model]['R2']
                models_data.append(model_r2)
                labels.append(model)

            if models_data:
                axes[1, 0].boxplot(models_data, labels=labels, patch_artist=True)
                axes[1, 0].set_title('R2 Score Distribution by Model', fontweight='bold')
                axes[1, 0].set_ylabel('R2 Score')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Distribution Data', ha='center', va='center',
                                transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Model Distribution Data', ha='center', va='center',
                            transform=axes[1, 0].transAxes)

        axes[1, 1].axis('off')
        if 'Model' in h3_overall.columns:
            model_stats = h3_overall.groupby('Model')['R2'].agg(['mean', 'std', 'min', 'max', 'count'])

            table_text = "Model Performance Statistics\n" + "=" * 40 + "\n"
            for model in model_stats.index:
                stats = model_stats.loc[model]
                table_text += f"{model}:\n"
                table_text += f"  Mean R2: {stats['mean']:.4f}\n"
                table_text += f"  Std Dev: {stats['std']:.4f}\n"
                table_text += f"  Min R2:  {stats['min']:.4f}\n"
                table_text += f"  Max R2:  {stats['max']:.4f}\n"
                table_text += f"  Count:   {int(stats['count'])}\n\n"

            best_combo = h3_overall.loc[h3_overall['R2'].idxmax()]
            table_text += "Best Performance:\n"
            table_text += f"  Model: {best_combo.get('Model', 'Unknown')}\n"
            table_text += f"  Horizon: {best_combo['Horizon_H']} days\n"
            table_text += f"  R2: {best_combo['R2']:.4f}\n"

            axes[1, 1].text(0.05, 0.95, table_text, transform=axes[1, 1].transAxes, fontsize=10,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            axes[1, 1].text(0.5, 0.5, 'No Model Statistics', ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        filename2 = f'{desktop_path}/02_H3_hypothesis_results.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()
        self.figures_saved.append(filename2)

    def create_plot3_market_regime_analysis(self):
        """Create market regime analysis visualization"""
        regime_results = []
        for result in self.h1_results + self.h3_results:
            if 'Regime' in result and result['Regime'] != 'Overall':
                regime_results.append(result)

        if not regime_results:
            print("Skipping Chart 3: No market regime data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Market Regime Stratification Analysis Results', fontsize=16, fontweight='bold')

        regime_df = pd.DataFrame(regime_results)

        regime_perf = regime_df.groupby('Regime')['R2'].agg(['mean', 'std', 'count'])
        bars = axes[0, 0].bar(regime_perf.index, regime_perf['mean'],
                              yerr=regime_perf['std'], capsize=5, alpha=0.7,
                              color=['#FFB6C1', '#98FB98', '#87CEEB'])
        axes[0, 0].set_title('Overall Performance by Market Regime', fontweight='bold')
        axes[0, 0].set_ylabel('Average R2 Score')
        axes[0, 0].grid(True, alpha=0.3)

        for bar, mean_val, count in zip(bars, regime_perf['mean'], regime_perf['count']):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{mean_val:.3f}\n(n={count})', ha='center', va='bottom', fontsize=10)

        h1_regime = regime_df[regime_df['Hypothesis'] == 'H1']
        if not h1_regime.empty and 'Model' in h1_regime.columns:
            model_regime_perf = h1_regime.groupby(['Model', 'Regime'])['R2'].mean().unstack()
            if not model_regime_perf.empty:
                model_regime_perf.plot(kind='bar', ax=axes[0, 1], alpha=0.7)
                axes[0, 1].set_title('H1: Model Performance by Market Regime', fontweight='bold')
                axes[0, 1].set_ylabel('R2 Score')
                axes[0, 1].legend(title='Market Regime')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No H1 Regime Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, 'No H1 Regime Data', ha='center', va='center', transform=axes[0, 1].transAxes)

        h3_regime = regime_df[regime_df['Hypothesis'] == 'H3']
        if not h3_regime.empty and 'Horizon_H' in h3_regime.columns:
            horizon_regime_perf = h3_regime.groupby(['Horizon_H', 'Regime'])['R2'].mean().unstack()
            if not horizon_regime_perf.empty:
                horizon_regime_perf.plot(kind='line', ax=axes[1, 0], marker='o')
                axes[1, 0].set_title('H3: Forecast Horizon Performance by Market Regime', fontweight='bold')
                axes[1, 0].set_xlabel('Forecast Horizon H (Days)')
                axes[1, 0].set_ylabel('R2 Score')
                axes[1, 0].legend(title='Market Regime')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No H3 Regime Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'No H3 Regime Data', ha='center', va='center', transform=axes[1, 0].transAxes)

        regime_stability = regime_df.groupby('Regime')['R2'].std()
        bars = axes[1, 1].bar(regime_stability.index, regime_stability.values, alpha=0.7,
                              color=['#FFB6C1', '#98FB98', '#87CEEB'])
        axes[1, 1].set_title('Market Regime Prediction Stability', fontweight='bold')
        axes[1, 1].set_ylabel('R2 Standard Deviation (Stability Index)')
        axes[1, 1].grid(True, alpha=0.3)

        for bar, std_val in zip(bars, regime_stability.values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{std_val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename3 = f'{desktop_path}/03_market_regime_analysis.png'
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close()
        self.figures_saved.append(filename3)

    def create_plot4_feature_importance(self):
        """Create feature importance analysis visualization"""
        if not self.feature_importance_data and not self.prediction_examples:
            print("Skipping Chart 4: No feature importance data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance and Prediction Performance Analysis', fontsize=16, fontweight='bold')

        if self.feature_importance_data and 'RF' in self.feature_importance_data:
            feature_names = ['Price', 'Log_Price', 'Return', 'Return_Lag1', 'Return_Lag2',
                             'SMA_15', 'SMA_45', 'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
                             'VFTSE', 'VFTSE_zscore']
            importance_scores = self.feature_importance_data['RF']

            min_len = min(len(feature_names), len(importance_scores))
            feature_names = feature_names[:min_len]
            importance_scores = importance_scores[:min_len]

            sorted_idx = np.argsort(importance_scores)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_scores = [importance_scores[i] for i in sorted_idx]

            top_features = sorted_features[:10]
            top_scores = sorted_scores[:10]

            bars = axes[0, 0].barh(range(len(top_features)), top_scores, alpha=0.7, color='lightgreen')
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features)
            axes[0, 0].set_xlabel('Importance Score')
            axes[0, 0].set_title('Feature Importance Ranking (Top 10)', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)

            for bar, score in zip(bars, top_scores):
                axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                                f'{score:.3f}', va='center', fontsize=9)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Feature Importance Data', ha='center', va='center',
                            transform=axes[0, 0].transAxes)

        if self.prediction_examples and 'y_true' in self.prediction_examples:
            y_true = self.prediction_examples['y_true']
            y_pred = self.prediction_examples['y_pred']
            model_name = self.prediction_examples['model_name']

            axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=30, color='blue')
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            r2_demo = r2_score(y_true, y_pred)
            axes[0, 1].set_xlabel('Actual Values')
            axes[0, 1].set_ylabel('Predicted Values')
            axes[0, 1].set_title(f'Prediction Performance Example - {model_name} (R2={r2_demo:.3f})', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Prediction Example Data', ha='center', va='center',
                            transform=axes[0, 1].transAxes)

        if self.prediction_examples and 'y_true' in self.prediction_examples:
            y_true = self.prediction_examples['y_true']
            y_pred = self.prediction_examples['y_pred']
            stock_name = self.prediction_examples['stock_name']

            time_points = range(len(y_true))
            axes[1, 0].plot(time_points, y_true, label='Actual Values', linewidth=2, color='blue')
            axes[1, 0].plot(time_points, y_pred, label='Predicted Values', linewidth=2, color='red', alpha=0.7)
            axes[1, 0].fill_between(time_points, y_true, y_pred, alpha=0.3, color='gray')

            axes[1, 0].set_xlabel('Time Points')
            axes[1, 0].set_ylabel('Return Rate')
            axes[1, 0].set_title(f'Time Series Prediction Performance - {stock_name[:10]}', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Time Series Data', ha='center', va='center', transform=axes[1, 0].transAxes)

        if self.prediction_examples and 'y_true' in self.prediction_examples:
            y_true = self.prediction_examples['y_true']
            y_pred = self.prediction_examples['y_pred']
            errors = y_true - y_pred

            axes[1, 1].hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {errors.mean():.4f}')
            axes[1, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Error')
            axes[1, 1].set_xlabel('Prediction Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Prediction Error Distribution', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Error Analysis Data', ha='center', va='center',
                            transform=axes[1, 1].transAxes)

        plt.tight_layout()
        filename4 = f'{desktop_path}/04_feature_importance_analysis.png'
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        plt.close()
        self.figures_saved.append(filename4)

    def create_plot5_comprehensive_summary(self):
        """Create comprehensive summary visualization"""
        all_results = []
        if self.h1_results:
            h1_overall = [r for r in self.h1_results if r.get('Regime', 'Overall') == 'Overall']
            for r in h1_overall:
                all_results.append({'Hypothesis': 'H1', 'R2': r['R2'], 'Type': r['Model']})
        if self.h2_results:
            for r in self.h2_results:
                all_results.append({'Hypothesis': 'H2', 'R2': r['R2'], 'Type': r['FeatureSet']})
        if self.h3_results:
            h3_overall = [r for r in self.h3_results if r.get('Regime', 'Overall') == 'Overall']
            for r in h3_overall:
                all_results.append(
                    {'Hypothesis': 'H3', 'R2': r['R2'], 'Type': f"{r.get('Model', 'Unknown')}_H={r['Horizon_H']}"})

        if not all_results:
            print("Skipping Chart 5: No experimental results data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Analysis Summary', fontsize=16, fontweight='bold')

        all_df = pd.DataFrame(all_results)

        hypothesis_perf = all_df.groupby('Hypothesis')['R2'].agg(['mean', 'std', 'count'])

        bars = axes[0, 0].bar(hypothesis_perf.index, hypothesis_perf['mean'],
                              yerr=hypothesis_perf['std'], capsize=5, alpha=0.7,
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Overall Performance Comparison of Three Hypotheses', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Average R2 Score', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)

        for bar, mean_val, count in zip(bars, hypothesis_perf['mean'], hypothesis_perf['count']):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{mean_val:.3f}\n(n={count})', ha='center', va='bottom', fontsize=10)

        total_stocks = len(self.all_stocks)
        analyzed_stocks = min(50, total_stocks)
        total_experiments = len(self.h1_results) + len(self.h2_results) + len(self.h3_results)

        coverage_data = [total_stocks, analyzed_stocks, total_experiments]
        coverage_labels = ['Total Stocks', 'Analyzed Stocks', 'Total Experiments']

        bars = axes[0, 1].bar(coverage_labels, coverage_data,
                              color=['skyblue', 'lightgreen', 'coral'], alpha=0.7)
        axes[0, 1].set_title('Data Coverage Analysis', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

        for bar, value in zip(bars, coverage_data):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                            str(value), ha='center', va='bottom', fontweight='bold')

        all_r2_values = [r['R2'] for r in all_results]

        axes[1, 0].hist(all_r2_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('R2 Distribution of All Experiments', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('R2 Score', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)

        mean_r2 = np.mean(all_r2_values)
        axes[1, 0].axvline(mean_r2, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_r2:.3f}')
        axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].axis('off')

        total_experiments = len(all_results)
        mean_r2 = np.mean(all_r2_values)
        max_r2 = np.max(all_r2_values)
        success_ratio = len([r for r in all_r2_values if r > 0]) / len(all_r2_values) * 100

        summary_text = """H1-H3 Hypothesis Stock Forecast Research Summary
====================
Experiment Scale:
- Total Stocks: {len(self.all_stocks)}
- H1 Experiments: {len(self.h1_results)}
- H2 Experiments: {len(self.h2_results)}
- H3 Experiments: {len(self.h3_results)}
- Total Experiments: {total_experiments}

Key Findings:
- Average R2: {mean_r2:.4f}
- Best R2: {max_r2:.4f}
- Success Rate: {success_ratio:.1f}%
- Total Errors: {len(self.error_log)}

Hypothesis Validation:
H1: Model Comparison + Market Regime
H2: Feature Engineering
H3: Multi-Model Forecast Horizon + Market Regime
Feature Importance Analysis
Prediction Visualization

H3 Enhanced Features:
SVR vs RF vs MLP Comparison
Multi-Horizon Analysis
Market Regime Stratification"""

        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, fontsize=9,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

        plt.tight_layout()
        filename5 = f'{desktop_path}/05_comprehensive_summary.png'
        plt.savefig(filename5, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        self.figures_saved.append(filename5)

    def create_all_plots(self):
        """Create all visualizations with available data"""
        print("\nGenerating H1-H3 hypothesis analysis plots...")

        try:
            self.create_plot1_h1h2_results()
            print("Chart 1: H1-H2 hypothesis results")
        except Exception as e:
            print(f"Chart 1 generation failed: {e}")

        try:
            self.create_plot2_h3_results()
            print("Chart 2: H3 hypothesis results")
        except Exception as e:
            print(f"Chart 2 generation failed: {e}")

        try:
            self.create_plot3_market_regime_analysis()
            print("Chart 3: Market regime analysis")
        except Exception as e:
            print(f"Chart 3 skipped or failed: {e}")

        try:
            self.create_plot4_feature_importance()
            print("Chart 4: Feature importance analysis")
        except Exception as e:
            print(f"Chart 4 generation failed: {e}")

        try:
            self.create_plot5_comprehensive_summary()
            print("Chart 5: Comprehensive analysis summary")
        except Exception as e:
            print(f"Chart 5 skipped or failed: {e}")

        print(f"Generated charts: {len(self.figures_saved)}")
        for fig in self.figures_saved:
            print(f"{os.path.basename(fig)}")

    def save_results(self):
        """Save all research results"""
        print("\nSaving H1-H3 hypothesis research results...")

        if self.h1_results:
            h1_df = pd.DataFrame(self.h1_results)
            h1_df.to_csv(f'{desktop_path}/H1_model_comparison_results.csv', index=False, encoding='utf-8-sig')
            print(f"H1 results: H1_model_comparison_results.csv")

        if self.h2_results:
            h2_df = pd.DataFrame(self.h2_results)
            h2_df.to_csv(f'{desktop_path}/H2_feature_engineering_results.csv', index=False, encoding='utf-8-sig')
            print(f"H2 results: H2_feature_engineering_results.csv")

        if self.h3_results:
            h3_df = pd.DataFrame(self.h3_results)
            h3_df.to_csv(f'{desktop_path}/H3_forecast_horizon_results.csv', index=False, encoding='utf-8-sig')
            print(f"H3 results: H3_forecast_horizon_results.csv")

        if self.error_log:
            with open(f'{desktop_path}/error_log.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.error_log))
            print(f"Error log: error_log.txt ({len(self.error_log)} entries)")

        all_results = []
        if self.h1_results:
            h1_overall = [r for r in self.h1_results if r.get('Regime', 'Overall') == 'Overall']
            for r in h1_overall:
                all_results.append({'Hypothesis': 'H1', 'R2': r['R2'], 'Type': r['Model']})
        if self.h2_results:
            for r in self.h2_results:
                all_results.append({'Hypothesis': 'H2', 'R2': r['R2'], 'Type': r['FeatureSet']})
        if self.h3_results:
            h3_overall = [r for r in self.h3_results if r.get('Regime', 'Overall') == 'Overall']
            for r in h3_overall:
                all_results.append(
                    {'Hypothesis': 'H3', 'R2': r['R2'], 'Type': f"{r.get('Model', 'Unknown')}_H={r['Horizon_H']}"})

        total_experiments = len(all_results)
        all_r2_values = [r['R2'] for r in all_results]

        report_lines = [
            "H1-H3 Hypothesis Stock Forecast Research Report",
            "=" * 80,
            f"Research Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data File: {os.path.basename(data_path)}",
            "",
            "Research Hypotheses:",
            "H1: Model Comparison (SVR vs RF vs MLP) + Market Regime Stratification",
            "H2: Feature Engineering (Price vs Price+Technical Indicators)",
            "H3: Forecast Horizon + Market Regime + Multi-Model Comparison",
            "",
            "Experiment Scale:",
            f"Total Stocks: {len(self.all_stocks)}",
            f"H1 Experiments: {len(self.h1_results)}",
            f"H2 Experiments: {len(self.h2_results)}",
            f"H3 Experiments: {len(self.h3_results)}",
            f"Total Experiments: {total_experiments}",
            f"Error Records: {len(self.error_log)}",
            "",
            "Key Findings:",
            f"Average R2: {np.mean(all_r2_values):.4f}" if all_r2_values else "Average R2: N/A",
            f"Best R2: {np.max(all_r2_values):.4f}" if all_r2_values else "Best R2: N/A",
            f"Success Rate: {len([r for r in all_r2_values if r > 0]) / len(all_r2_values) * 100:.1f}%" if all_r2_values else "Success Rate: N/A",
            "",
            "Output Files:",
            f"Charts: {len(self.figures_saved)}"
        ]

        for fig in self.figures_saved:
            report_lines.append(f"{os.path.basename(fig)}")

        report_lines.extend([
            f"Data Files: 3 (H1-H3 results CSV)",
            f"Log File: 1 (error log)",
            "",
            "Academic Value:",
            "Constructed reproducible benchmark evaluation framework for UK market",
            "Revealed utility boundaries of feature extension and model stability issues",
            "Provided foundation for introducing multi-source heterogeneous information",
            "",
            "Practical Implications:",
            "Suggested viewing ML outputs as auxiliary tools for risk management",
            "Positioned models for market state recognition rather than direct trading signals",
            "Provided reference for actual financial decision-making needs"
        ])

        with open(f'{desktop_path}/H1H3_research_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"Comprehensive report: H1H3_research_report.txt")


def main():
    """Main function for H1-H3 hypothesis research system"""
    print("=" * 90)
    print("Enhanced H1-H3 Hypothesis Stock Forecast Research System")
    print("H1: Model Comparison + Market Regime")
    print("H2: Feature Engineering")
    print("H3: Forecast Horizon + Market Regime + Multi-Model Comparison")
    print("=" * 90)

    if data_path is None:
        print(f"Please ensure CSV files exist in: {data_folder}")
        return

    research = StockForecastResearchSystem()

    start_time = pd.Timestamp.now()

    try:
        print("\nData loading phase")
        if not research.load_stock_data(data_path):
            print("Data loading failed, program exiting")
            return

        print("\nStarting H1-H3 hypothesis validation")

        print("\n" + "=" * 50)
        research.conduct_h1_hypothesis(sample_stocks=50)

        print("\n" + "=" * 50)
        research.conduct_h2_hypothesis(sample_stocks=30)

        print("\n" + "=" * 50)
        research.conduct_h3_hypothesis(sample_stocks=30)

        print("\n" + "=" * 50)
        research.create_all_plots()

        print("\n" + "=" * 50)
        research.save_results()

        end_time = pd.Timestamp.now()
        total_time = (end_time - start_time).total_seconds() / 60

        print(f"\n{'=' * 90}")
        print(f"Enhanced H1-H3 Hypothesis Research System completed!")
        print(f"{'=' * 90}")
        print(f"Execution Statistics:")
        print(f"Total time: {total_time:.1f} minutes")
        print(f"Analyzed stocks: {len(research.all_stocks)}")
        print(f"H1 experiments: {len(research.h1_results)} (including stratification)")
        print(f"H2 experiments: {len(research.h2_results)}")
        print(f"H3 experiments: {len(research.h3_results)} (including multi-model comparison)")
        print(f"Total experiments: {total_experiments}")
        print(f"Error records: {len(research.error_log)}")
        print(f"Generated charts: {len(research.figures_saved)}")
        print(f"Saved files: {len(research.figures_saved) + 4}")

        print(f"\nH3 Enhanced Features:")
        print(f"SVR vs RF vs MLP full comparison")
        print(f"Multi-horizon analysis (H=5,10,30,60 days)")
        print(f"Market regime stratification")
        print(f"Model performance heatmap visualization")

        print(f"\nSystem Enhancements:")
        print(f"Complete error handling mechanism")
        print(f"Strict time series validation")
        print(f"Feature importance analysis")
        print(f"Prediction visualization")

        print(f"\nHypothesis Validation:")
        print(f"H1: Model Comparison + Cross-market-state verification")
        print(f"H2: Feature Engineering complete evaluation")
        print(f"H3: Multi-model forecast horizon + Cross-market-state verification")

        print(f"\nChart Files:")
        for i, fig_path in enumerate(research.figures_saved, 1):
            print(f"{i:02d}. {os.path.basename(fig_path)}")

    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()

        try:
            research.save_results()
            print("Partial results saved")
        except:
            print("Result saving also failed")


if __name__ == "__main__":
    main()