import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
from fetch_data import forex_data
from model import ForexModelTrainer
from fetch_data import forex_data

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ForexVisualizer:
    def __init__(self, trainer, X_test, y_test, results):
        self.trainer = trainer
        self.X_test = X_test
        self.y_test = y_test
        self.results = results
        
    def plot_predictions_vs_actual(self, model_name='Random Forest', n_points=200):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        y_pred = self.results[model_name]['predictions']
        
        # plot last n_points
        plot_range = slice(-n_points, None)
        test_dates = self.y_test.index[plot_range]
        
        # time series plot
        ax1.plot(test_dates, self.y_test.iloc[plot_range], 
                label='Actual Returns', color='blue', alpha=0.7, linewidth=2)
        ax1.plot(test_dates, y_pred[plot_range], 
                label='Predicted Returns', color='red', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Return', fontsize=12)
        ax1.set_title(f'{model_name}: Predicted vs Actual Returns (Last {n_points} days)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # scatter plot
        ax2.scatter(self.y_test, y_pred, alpha=0.5, s=20)
        
        # add diagonal line (perfect predictions)
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Returns', fontsize=12)
        ax2.set_ylabel('Predicted Returns', fontsize=12)
        ax2.set_title(f'{model_name}: Prediction Scatter Plot', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_predictions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        models = list(self.results.keys())
        metrics = ['RMSE', 'MAE', 'R2']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            values = [self.results[model][metric] for model in models]
            axes[idx].bar(models, values, color=color, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v, f'{v:.4f}', 
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model_name='Random Forest', top_n=20):
        importance_df = self.trainer.get_feature_importance(model_name, top_n)
        
        if importance_df is None:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['importance'], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_residuals(self, model_name='Random Forest'):
        y_pred = self.results[model_name]['predictions']
        residuals = self.y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # residuals over time
        axes[0, 0].plot(self.y_test.index, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Date', fontsize=12)
        axes[0, 0].set_ylabel('Residuals', fontsize=12)
        axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # residuals vs predictions
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Returns', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # histogram of residuals
        axes[1, 0].hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{model_name}: Residual Analysis', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_residuals.png', dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_error_distribution(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, result) in enumerate(self.results.items()):
            y_pred = result['predictions']
            errors = np.abs(self.y_test - y_pred)
            
            axes[idx].hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('Absolute Error', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{name}\nMAE: {result["MAE"]:.6f}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle('Error Distribution Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_cumulative_returns(self, model_name='Random Forest'):
        y_pred = self.results[model_name]['predictions']
        
        # calculate cumulative returns
        actual_cumulative = (1 + self.y_test).cumprod()
        
        # strategy: go long if prediction is positive, short if negative
        predicted_direction = np.sign(y_pred)
        strategy_returns = predicted_direction * self.y_test
        strategy_cumulative = (1 + strategy_returns).cumprod()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(self.y_test.index, actual_cumulative, label='Buy and Hold', linewidth=2, color='blue')
        ax.plot(self.y_test.index, strategy_cumulative, label='ML Strategy', linewidth=2, color='green')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.set_title(f'{model_name}: Cumulative Returns - Buy&Hold vs ML Strategy', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # add final returns as text
        final_actual = actual_cumulative.iloc[-1] - 1
        final_strategy = strategy_cumulative.iloc[-1] - 1
        
        textstr = f'Buy & Hold Return: {final_actual:.2%}\nML Strategy Return: {final_strategy:.2%}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.show()
    

    def plot_price_prediction(self, model_name='Random Forest', n_days=100):
        y_pred = self.results[model_name]['predictions']
        
        # get actual prices from test set
        plot_range = slice(-n_days, None)
        dates = self.y_test.index[plot_range]
        actual_returns = self.y_test.iloc[plot_range]
        pred_returns = y_pred[plot_range]
        
        # get the starting price (from original data)
        start_idx = self.y_test.index.get_loc(dates[0])
        original_data = forex_data[self.trainer.target_pair]
        start_price = original_data.loc[dates[0], 'Close']
        
        # reconstruct price series
        actual_prices = [start_price]
        pred_prices = [start_price]
        
        for i in range(len(actual_returns)):
            actual_prices.append(actual_prices[-1] * (1 + actual_returns.iloc[i]))
            pred_prices.append(pred_prices[-1] * (1 + pred_returns[i]))
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(dates, actual_prices[1:], label='Actual Price', linewidth=2, color='blue', alpha=0.8)
        ax.plot(dates, pred_prices[1:], label='Predicted Price', linewidth=2, color='red', alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.set_title(f'{model_name}: Actual vs Predicted Price (Last {n_days} days)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_price_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
    

    def create_summary_report(self):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # model comparison
        ax1 = fig.add_subplot(gs[0, :])
        models = list(self.results.keys())
        rmse_vals = [self.results[m]['RMSE'] for m in models]
        mae_vals = [self.results[m]['MAE'] for m in models]
        r2_vals = [self.results[m]['R2'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, rmse_vals, width, label='RMSE', alpha=0.8)
        ax1.bar(x, mae_vals, width, label='MAE', alpha=0.8)
        ax1.bar(x + width, r2_vals, width, label='R²', alpha=0.8)
        
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # best model predictions
        best_model = min(self.results, key=lambda x: self.results[x]['RMSE'])
        y_pred = self.results[best_model]['predictions']
        
        ax2 = fig.add_subplot(gs[1, :2])
        plot_range = slice(-200, None)
        ax2.plot(self.y_test.index[plot_range], self.y_test.iloc[plot_range], label='Actual', linewidth=2, alpha=0.8)
        ax2.plot(self.y_test.index[plot_range], y_pred[plot_range], label='Predicted', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Returns', fontsize=11)
        ax2.set_title(f'Best Model ({best_model}): Predictions', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # scatter plot
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.scatter(self.y_test, y_pred, alpha=0.4, s=10)
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax3.set_xlabel('Actual', fontsize=11)
        ax3.set_ylabel('Predicted', fontsize=11)
        ax3.set_title('Prediction Scatter', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # residuals
        ax4 = fig.add_subplot(gs[2, 0])
        residuals = self.y_test - y_pred
        ax4.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Residuals', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.grid(True, alpha=0.3)
        
        # feature importance (if available)
        ax5 = fig.add_subplot(gs[2, 1:])
        importance_df = self.trainer.get_feature_importance(best_model, top_n=10)
        if importance_df is not None:
            y_pos = np.arange(len(importance_df))
            ax5.barh(y_pos, importance_df['importance'], color='steelblue', alpha=0.8)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(importance_df['feature'], fontsize=9)
            ax5.invert_yaxis()
            ax5.set_xlabel('Importance', fontsize=11)
            ax5.set_title(f'Top 10 Features - {best_model}', 
                         fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(f'Forex Prediction Model Summary - {self.trainer.target_pair}', 
                    fontsize=18, fontweight='bold')
        
        plt.savefig('visualizations/summary_report.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    import os
    
    # create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    print("="*60)
    print("FOREX PREDICTION VISUALIZATION")
    print("="*60)
    
    # Llad training results
    print("\nLoading training results...")
    with open('models/training_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    trainer = data['trainer']
    X_test = data['X_test']
    y_test = data['y_test']
    results = data['results']
    
    print(f"Loaded results for {trainer.target_pair}")
    print(f"Test set size: {len(X_test)}")
    print(f"Models: {list(results.keys())}")
    
    # create visualizer
    viz = ForexVisualizer(trainer, X_test, y_test, results)
    
    # generate all visualizations
    print("\nGenerating visualizations...")
    
    print("\n1. Creating summary report...")
    viz.create_summary_report()
    
    print("\n2. Plotting predictions vs actual...")
    viz.plot_predictions_vs_actual('Random Forest')
    
    print("\n3. Plotting model comparison...")
    viz.plot_model_comparison()
    
    print("\n4. Plotting feature importance...")
    viz.plot_feature_importance('Random Forest', top_n=20)
    
    print("\n5. Plotting residual analysis...")
    viz.plot_residuals('Random Forest')
    
    print("\n6. Plotting error distribution...")
    viz.plot_error_distribution()
    
    print("\n7. Plotting cumulative returns...")
    viz.plot_cumulative_returns('Random Forest')
    
    print("\n8. Plotting price predictions...")
    viz.plot_price_prediction('Random Forest', n_days=100)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("All plots saved in 'visualizations/' directory")
    print("="*60)