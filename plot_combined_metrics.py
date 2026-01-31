#!/usr/bin/env python3
"""
Script to create a combined figure with three scatter plots showing 
model release dates vs different performance metrics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import numpy as np
from datetime import datetime

# File paths
JSON_FILE = '/data/ztw/simple-mmeval-infer-data/MODEL_CONFIGS_updated.json'
CSV_FILE = '/data/ztw/simple-mmeval-infer-data/conservation_all_strict_vs_average_results.csv'
OUTPUT_DIR = '/data/ztw/simple-mmeval-infer-data/'

# Three metrics to plot (excluding strict)
METRICS = [
    {
        'column': 'Average_Accuracy',
        'title': 'Average Accuracy',
        'position': 2
    },
    {
        'column': 'Main_Accuracy',
        'title': 'Conservation Task Accuracy',
        'position': 0
    },
    {
        'column': 'Control_Accuracy',
        'title': 'Non-Conserving Control Task Accuracy',
        'position': 1
    }
]

def load_model_configs(json_file):
    """Load model configurations from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_performance_data(csv_file):
    """Load performance data from CSV file."""
    df = pd.read_csv(csv_file)
    return df

def match_models(model_configs, performance_df, metric_columns):
    """
    Match models between JSON and CSV files for multiple metrics.
    Returns matched data and list of skipped models.
    """
    matched_data = []
    skipped_models = []
    
    # Iterate through each model in the CSV
    for _, row in performance_df.iterrows():
        model_name = row['Model']
        
        # Try to find the model in the JSON configs
        if model_name in model_configs:
            created_at = model_configs[model_name].get('created_at')
            parameter_count = model_configs[model_name].get('parameter_count', 0)
            
            if created_at:
                try:
                    # Parse the date
                    date_obj = datetime.strptime(created_at, '%Y-%m-%d')
                    
                    # Get all metric values
                    data_point = {
                        'model': model_name,
                        'date': date_obj,
                        'params': parameter_count
                    }
                    
                    # Add each metric value
                    for col in metric_columns:
                        data_point[col] = row[col]
                    
                    matched_data.append(data_point)
                    
                except ValueError:
                    skipped_models.append(f"{model_name} (invalid date format: {created_at})")
            else:
                skipped_models.append(f"{model_name} (no created_at field)")
        else:
            skipped_models.append(f"{model_name} (not found in JSON)")
    
    return matched_data, skipped_models

def create_combined_plot(matched_data, metrics):
    """Create a combined figure with three subplots arranged horizontally."""
    
    # Sort by date for better visualization
    matched_data.sort(key=lambda x: x['date'])
    
    # Extract common data
    dates = [d['date'] for d in matched_data]
    params = [d['params'] for d in matched_data]
    dates_numeric = mdates.date2num(dates)
    
    # Normalize parameter counts for color mapping (log scale)
    params_positive = [p if (p is not None and p > 0) else 1e6 for p in params]
    params_log = np.log10(params_positive)
    min_param_log = min(params_log)
    max_param_log = max(params_log)
    norm_params = [(p - min_param_log) / (max_param_log - min_param_log) for p in params_log]
    
    # Use GnBu colormap
    cmap = plt.cm.GnBu
    
    # Determine global y-axis limits
    all_accuracies = []
    for metric in metrics:
        all_accuracies.extend([d[metric['column']] for d in matched_data])
    y_max_global = max(all_accuracies)
    y_lim = (0, min(1.0, y_max_global * 1.05))
    
    # Create figure with 3 subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))
    
    # Process each metric
    for metric in metrics:
        ax = axes[metric['position']]
        metric_column = metric['column']
        title = metric['title']
        
        # Extract accuracies for this metric
        accuracies = [d[metric_column] for d in matched_data]
        
        # Create scatter plot with colors based on model size
        for i, (date, acc, norm_p) in enumerate(zip(dates, accuracies, norm_params)):
            color_val = cmap(norm_p)
            ax.scatter(date, acc, s=100, alpha=0.8, color=color_val,
                      edgecolor='black', linewidth=1, zorder=3)
        
        # Add regression line
        coeffs = np.polyfit(dates_numeric, accuracies, 1)
        poly = np.poly1d(coeffs)
        dates_line = np.linspace(min(dates_numeric), max(dates_numeric), 100)
        accuracies_line = poly(dates_line)
        
        ax.plot([mdates.num2date(d) for d in dates_line], accuracies_line, 
               '--', color='#e6953d', linewidth=1.5, 
               label='Regression Line', zorder=2)
        
        # Customize subplot
        ax.set_xlabel('Model Release Date', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=20, pad=20)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Set y-axis limits (shared across all subplots)
        ax.set_ylim(y_lim)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=14, frameon=True)
        
        # Only add y-label to the leftmost subplot
        if metric['position'] == 0:
            ax.set_ylabel('Model Performance (Accuracy)', fontsize=18, fontweight='bold')
    
    # Adjust layout to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    # Add shared colorbar on the right side
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                               norm=Normalize(vmin=min_param_log, vmax=max_param_log))
    sm.set_array([])
    
    # Create colorbar axis
    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Model Size (log10 Parameters)', fontsize=16)
    
    return fig

def main():
    """Main execution function."""
    print("="*70)
    print("Model Performance vs Release Date - Combined Three Metrics")
    print("="*70)
    
    print("\nLoading model configurations...")
    model_configs = load_model_configs(JSON_FILE)
    print(f"Loaded {len(model_configs)} model configurations")
    
    print("\nLoading performance data...")
    performance_df = load_performance_data(CSV_FILE)
    print(f"Loaded performance data for {len(performance_df)} models")
    
    # Get all metric columns
    metric_columns = [m['column'] for m in METRICS]
    
    print("\nMatching models between datasets...")
    matched_data, skipped_models = match_models(model_configs, performance_df, metric_columns)
    print(f"Successfully matched {len(matched_data)} models")
    
    if skipped_models:
        print(f"\nSkipped {len(skipped_models)} models:")
        for model in skipped_models:
            print(f"  - {model}")
    
    if not matched_data:
        print("\nError: No models matched between datasets!")
        return
    
    print("\nCreating combined scatter plot with 3 metrics...")
    fig = create_combined_plot(matched_data, METRICS)
    
    # Save PDF
    pdf_path = OUTPUT_DIR + 'model_performance_combined_three_metrics.pdf'
    print(f"\nSaving plot to {pdf_path}...")
    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print("PDF plot saved successfully!")
    
    # Save PNG
    png_path = OUTPUT_DIR + 'model_performance_combined_three_metrics.png'
    print(f"Saving plot to {png_path}...")
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print("PNG plot saved successfully!")
    
    # Display statistics
    print(f"\n{'='*70}")
    print("Statistics for each metric:")
    print(f"{'='*70}")
    
    for metric in METRICS:
        col = metric['column']
        accuracies = [d[col] for d in matched_data]
        print(f"\n{metric['title']}:")
        print(f"  Range: {min(accuracies):.4f} to {max(accuracies):.4f}")
        print(f"  Mean: {np.mean(accuracies):.4f}")
    
    # Show date and parameter ranges
    dates = [d['date'] for d in matched_data]
    params = [d['params'] for d in matched_data]
    params_positive = [p for p in params if p is not None and p > 0]
    
    print(f"\nDate range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
    if params_positive:
        print(f"Model size range: {min(params_positive):.2e} to {max(params_positive):.2e} parameters")
    
    print(f"\n{'='*70}")
    print("All plots generated successfully!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

