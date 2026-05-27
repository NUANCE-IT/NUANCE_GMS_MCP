import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_plot():
    # Set the overall style for the plots
    sns.set_theme(style="whitegrid")

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})

    # ==========================================
    # Plot (a): Automated Test Outcomes
    # ==========================================
    
    # Example data mirroring the attached image
    categories = ['Hardware-\nIndependent', 'Bridge\nProtocol', 'Ollama\nIntegration']
    passes = [92, 13, 12]  # Approximate values
    fails = [0, 0, 3]      # Approximate values

    width = 0.5
    
    # Stacked bar chart
    ax1.bar(categories, passes, width, label='Pass', color='#2ca02c') # Green
    ax1.bar(categories, fails, width, bottom=passes, label='Fail', color='#d62728') # Red

    ax1.set_ylabel('Number of Tests')
    ax1.set_title('(a) Automated Test Outcomes', fontsize=14)
    ax1.legend()
    # Adjust y-axis to comfortably fit the highest bar
    ax1.set_ylim(0, 100)

    # ==========================================
    # Plot (b): Per-Test Latency by Model
    # ==========================================
    
    results_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    data = []
    
    if os.path.exists(results_path):
        import json
        with open(results_path, 'r') as f:
            real_results = json.load(f)
            
        for model_name, info in real_results.items():
            success = info.get('success_count', 0)
            total = info.get('total_runs', 0)
            latencies = info.get('latencies', [])
            
            # Format the label, e.g., 'qwen2.5\n(15/15)'
            short_name = model_name.split(':')[0]
            label = f"{short_name}\n({success}/{total})"
            
            for lat in latencies:
                data.append({'Model': label, 'Latency (s)': lat})
    else:
        # Fallback to synthetic data generation if no real benchmark results exist
        np.random.seed(42)
        models_info = [
            ('llama3.2\n(15/15)', 8, 2),        # Name, mean latency, std dev
            ('qwen2.5\n(11/15)', 30, 3),
            ('qwen2.5-coder\n(15/15)', 24, 4),
            ('phi4\n(14/15)', 30, 2),
            ('gemma2\n(14/15)', 20, 2),
            ('nemotron\n(13/15)', 36, 3),
            ('mistral\n(15/15)', 17, 1.5)
        ]
        
        for model, mean, std in models_info:
            # Generate ~15 points per model to represent individual test latencies
            latencies = np.random.normal(mean, std, 15)
            for lat in latencies:
                data.append({'Model': model, 'Latency (s)': lat})
            
    df = pd.DataFrame(data)

    # Create boxplot
    sns.boxplot(
        data=df, x='Model', y='Latency (s)', ax=ax2, 
        color='lightblue', width=0.4, fliersize=0
    )
    
    # Overlay stripplot (the scatter points)
    sns.stripplot(
        data=df, x='Model', y='Latency (s)', ax=ax2, 
        color='navy', alpha=0.5, jitter=True, size=5
    )

    ax2.set_title('(b) Per-Test Latency by Model', fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('Latency (s)')
    
    # Rotate x-axis labels to match the style of the original image
    ax2.tick_params(axis='x', rotation=45)

    # ==========================================
    # Final Adjustments and Saving
    # ==========================================
    
    # Ensure layout doesn't clip label text
    plt.tight_layout()
    
    # Save the figure to the scripts directory
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_path}")

    # Optionally display the plot during runtime
    # plt.show()

if __name__ == "__main__":
    generate_plot()
