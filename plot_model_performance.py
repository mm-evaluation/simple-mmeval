#!/usr/bin/env python3
"""
Script to create scatter plots showing model release dates vs various performance metrics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from datetime import datetime

# File paths
JSON_FILE = '/data/ztw/simple-mmeval-infer-data/MODEL_CONFIGS_updated.json'
CSV_FILE = '/data/ztw/simple-mmeval-infer-data/conservation_all_strict_vs_average_results.csv'
OUTPUT_DIR = '/data/ztw/simple-mmeval-infer-data/'

# Metrics to plot
METRICS = {
    'average': {
        'column': 'Average_Accuracy',
        'ylabel': 'Model Performance (Average Accuracy)',
        'title': 'Model Performance vs Release Date',
        'filename': 'model_performance_average_vs_release_date'
    },
    'strict': {
        'column': 'Strict_Logical_AND',
        'ylabel': 'Model Performance (Strict Accuracy)',
        'title': 'Model Performance (Strict) vs Release Date',
        'filename': 'model_performance_strict_vs_release_date'
    },
    'conservation': {
        'column': 'Main_Accuracy',
        'ylabel': 'Model Performance (Conservation Accuracy)',
        'title': 'Model Performance (Conservation) vs Release Date',
        'filename': 'model_performance_conservation_vs_release_date'
    },
    'non_conservation': {
        'column': 'Control_Accuracy',
        'ylabel': 'Model Performance (Non-Conservation Accuracy)',
        'title': 'Model Performance (Non-Conservation) vs Release Date',
        'filename': 'model_performance_non_conservation_vs_release_date'
    }
}

def load_model_configs(json_file):
    """Load model configurations from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_performance_data(csv_file):
    """Load performance data from CSV file."""
    df = pd.read_csv(csv_file)
    return df

def match_models(model_configs, performance_df, metric_column):
    """
    Match models between JSON and CSV files for a specific metric.
    Returns matched data and list of skipped models.
    """
    matched_data = []
    skipped_models = []
    
    # Iterate through each model in the CSV
    for _, row in performance_df.iterrows():
        model_name = row['Model']
        metric_value = row[metric_column]
        
        # Try to find the model in the JSON configs
        if model_name in model_configs:
            created_at = model_configs[model_name].get('created_at')
            parameter_count = model_configs[model_name].get('parameter_count', 0)
            
            if created_at:
                try:
                    # Parse the date
                    date_obj = datetime.strptime(created_at, '%Y-%m-%d')
                    matched_data.append({
                        'model': model_name,
                        'date': date_obj,
                        'accuracy': metric_value,
                        'params': parameter_count
                    })
                except ValueError:
                    skipped_models.append(f"{model_name} (invalid date format: {created_at})")
            else:
                skipped_models.append(f"{model_name} (no created_at field)")
        else:
            skipped_models.append(f"{model_name} (not found in JSON)")
    
    return matched_data, skipped_models

def create_plot(matched_data, ylabel, title):
    """Create a professional academic-style scatter plot with gradient coloring and regression line."""
    # Sort by date for better visualization
    matched_data.sort(key=lambda x: x['date'])
    
    # Extract dates, accuracies, and parameters
    dates = [d['date'] for d in matched_data]
    accuracies = [d['accuracy'] for d in matched_data]
    params = [d['params'] for d in matched_data]
    
    # Convert dates to numerical values for regression
    dates_numeric = mdates.date2num(dates)
    
    # Create figure with size matching reference code
    plt.figure(figsize=(12, 10))
    
    # Normalize parameter counts for color mapping (log scale for better visualization)
    # Filter out zero, negative, or None params for log scale
    params_positive = [p if (p is not None and p > 0) else 1e6 for p in params]  # Use 1M as default for missing values
    params_log = np.log10(params_positive)
    
    # Normalize for color mapping
    min_param_log = min(params_log)
    max_param_log = max(params_log)
    norm_params = [(p - min_param_log) / (max_param_log - min_param_log) for p in params_log]
    
    # Use GnBu colormap (Green-Blue) matching reference code
    cmap = plt.cm.GnBu
    
    # Create scatter plot with colors based on model size
    for i, (date, acc, norm_p) in enumerate(zip(dates, accuracies, norm_params)):
        color_val = cmap(norm_p)
        plt.scatter(date, acc, s=100, alpha=0.8, color=color_val,
                   edgecolor='black', linewidth=1, zorder=3)
    
    # Add regression line
    # Perform linear regression
    coeffs = np.polyfit(dates_numeric, accuracies, 1)
    poly = np.poly1d(coeffs)
    
    # Generate points for the regression line
    dates_line = np.linspace(min(dates_numeric), max(dates_numeric), 100)
    accuracies_line = poly(dates_line)
    
    # Plot regression line with orange color from reference code
    plt.plot([mdates.num2date(d) for d in dates_line], accuracies_line, 
            '--', color='#e6953d', linewidth=1.5, 
            label='Regression Line', zorder=2)
    
    # Customize axes with font sizes matching reference code
    plt.xlabel('Model Release Date', fontsize=18, fontweight='bold')
    plt.ylabel(ylabel, fontsize=18, fontweight='bold')
    plt.title(title, fontsize=20, pad=20)
    
    # Format x-axis to show dates nicely
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add grid matching reference code style
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Set y-axis range to show full scale from 0 to 1 (or slightly above max if needed)
    y_max = max(accuracies)
    plt.ylim(0, min(1.0, y_max * 1.05))
    
    # Add colorbar to show model size scale
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=Normalize(vmin=min_param_log, vmax=max_param_log))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Model Size (log10 Parameters)', fontsize=16)
    
    # Add legend for regression line
    plt.legend(loc='lower right', fontsize=14, frameon=True)
    
    # Improve layout
    plt.tight_layout()
    
    return plt

def generate_plot_for_metric(model_configs, performance_df, metric_key, metric_config):
    """Generate and save plots for a specific metric."""
    print(f"\n{'='*60}")
    print(f"Processing metric: {metric_key.upper()}")
    print(f"{'='*60}")
    
    metric_column = metric_config['column']
    ylabel = metric_config['ylabel']
    title = metric_config['title']
    filename = metric_config['filename']
    
    print(f"Matching models for {metric_column}...")
    matched_data, skipped_models = match_models(model_configs, performance_df, metric_column)
    print(f"Successfully matched {len(matched_data)} models")
    
    if skipped_models and metric_key == 'average':  # Only print once
        print(f"\nSkipped {len(skipped_models)} models:")
        for model in skipped_models:
            print(f"  - {model}")
    
    if not matched_data:
        print(f"\nError: No models matched for {metric_column}!")
        return
    
    print(f"Creating scatter plot...")
    plt_obj = create_plot(matched_data, ylabel, title)
    
    # Save PDF
    pdf_path = OUTPUT_DIR + filename + '.pdf'
    print(f"Saving plot to {pdf_path}...")
    plt_obj.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print("PDF plot saved successfully!")
    
    # Save PNG
    png_path = OUTPUT_DIR + filename + '.png'
    print(f"Saving plot to {png_path}...")
    plt_obj.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print("PNG plot saved successfully!")
    
    # Close the figure to free memory
    plt.close()
    
    # Display statistics
    accuracies = [d['accuracy'] for d in matched_data]
    print(f"\n--- {metric_key.upper()} Statistics ---")
    print(f"Accuracy range: {min(accuracies):.4f} to {max(accuracies):.4f}")

def main():
    """Main execution function."""
    print("="*60)
    print("Model Performance vs Release Date - Multiple Metrics")
    print("="*60)
    
    print("\nLoading model configurations...")
    model_configs = load_model_configs(JSON_FILE)
    print(f"Loaded {len(model_configs)} model configurations")
    
    print("\nLoading performance data...")
    performance_df = load_performance_data(CSV_FILE)
    print(f"Loaded performance data for {len(performance_df)} models")
    
    # Generate plots for each metric
    for metric_key, metric_config in METRICS.items():
        generate_plot_for_metric(model_configs, performance_df, metric_key, metric_config)
    
    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"{'='*60}")
    print(f"\nGenerated {len(METRICS)} plots (PDF + PNG for each):")
    for metric_key, metric_config in METRICS.items():
        print(f"  - {metric_config['filename']}.pdf")
        print(f"  - {metric_config['filename']}.png")

if __name__ == '__main__':
    main()
