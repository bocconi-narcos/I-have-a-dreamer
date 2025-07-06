#!/usr/bin/env python3
"""
ViTARC Comparison Runner
=======================

Utility script to run ViTARC positional embedding comparison experiments
with different configurations and options.
"""

import argparse
import os
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path

def run_experiments(subset=None, use_wandb=True, max_epochs=None):
    """Run the ViTARC comparison experiments"""
    
    # Setup environment variables
    env = os.environ.copy()
    if not use_wandb:
        env['WANDB_MODE'] = 'disabled'
    
    # Prepare command
    cmd = [sys.executable, 'train_ViTARC.py']
    
    # Add subset filter if specified
    if subset:
        print(f"Running subset: {subset}")
        # You can modify train_ViTARC.py to accept command line arguments
        # for now, we'll run the full set
    
    # Run the experiment
    print("Starting ViTARC comparison experiments...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        if result.returncode == 0:
            print("Experiments completed successfully!")
            return True
        else:
            print(f"Experiments failed with return code: {result.returncode}")
            return False
    except Exception as e:
        print(f"Error running experiments: {e}")
        return False

def analyze_results(results_dir="vitarc_experiments"):
    """Analyze the experimental results"""
    
    results_file = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Top performers
    print("\nTop 5 Configurations by Validation Accuracy:")
    top_configs = df.nlargest(5, 'best_val_acc')
    for i, row in top_configs.iterrows():
        print(f"{row['config_name']}: {row['best_val_acc']:.4f} accuracy (epoch {row.get('best_epoch', 'N/A')})")
        print(f"  Description: {row['description']}")
        print(f"  APE: {row['ape_type']}, RPE: {row['rpe_type']}, OPE: {row['use_OPE']}")
        print(f"  Time: {row['training_time']:.1f}s, Params: {row['num_parameters']:,}")
        
        # Early stopping info if available
        if 'early_stopped' in row and 'stopping_reason' in row:
            early_stop_status = "Early stopped" if row['early_stopped'] else "Completed training"
            convergence_eff = row.get('convergence_efficiency', 'N/A')
            print(f"  Training: {early_stop_status}, Convergence efficiency: {convergence_eff}")
            if row['early_stopped']:
                print(f"  Stopping reason: {row['stopping_reason']}")
        print()
    
    # Component analysis
    print("\nComponent Analysis:")
    
    # APE Type analysis
    ape_analysis = df.groupby('ape_type')['best_val_acc'].agg(['mean', 'max', 'count'])
    print("\nAPE Type Performance:")
    print(ape_analysis.round(4))
    
    # RPE Type analysis
    rpe_analysis = df.groupby('rpe_type')['best_val_acc'].agg(['mean', 'max', 'count'])
    print("\nRPE Type Performance:")
    print(rpe_analysis.round(4))
    
    # OPE analysis
    ope_analysis = df.groupby('use_OPE')['best_val_acc'].agg(['mean', 'max', 'count'])
    print("\nOPE Usage Performance:")
    print(ope_analysis.round(4))
    
    # Mixer strategy analysis
    mixer_analysis = df.groupby('ape_mixer_strategy')['best_val_acc'].agg(['mean', 'max', 'count'])
    print("\nMixer Strategy Performance:")
    print(mixer_analysis.round(4))
    
    # Training efficiency analysis
    print("\nTraining Efficiency:")
    print(f"Average training time: {df['training_time'].mean():.1f}s")
    print(f"Fastest training: {df['training_time'].min():.1f}s ({df.loc[df['training_time'].idxmin(), 'config_name']})")
    print(f"Slowest training: {df['training_time'].max():.1f}s ({df.loc[df['training_time'].idxmax(), 'config_name']})")
    
    # Parameter analysis
    print("\nModel Size Analysis:")
    print(f"Average parameters: {df['num_parameters'].mean():,.0f}")
    print(f"Smallest model: {df['num_parameters'].min():,} ({df.loc[df['num_parameters'].idxmin(), 'config_name']})")
    print(f"Largest model: {df['num_parameters'].max():,} ({df.loc[df['num_parameters'].idxmax(), 'config_name']})")
    
    # Early stopping analysis if available
    if 'early_stopped' in df.columns:
        print("\nEarly Stopping Analysis:")
        early_stopped_count = df['early_stopped'].sum()
        total_experiments = len(df)
        print(f"Experiments early stopped: {early_stopped_count}/{total_experiments} ({early_stopped_count/total_experiments*100:.1f}%)")
        
        if 'best_epoch' in df.columns:
            avg_best_epoch = df['best_epoch'].mean()
            print(f"Average best epoch: {avg_best_epoch:.1f}")
        
        if 'convergence_efficiency' in df.columns:
            avg_convergence_eff = df['convergence_efficiency'].mean()
            print(f"Average convergence efficiency: {avg_convergence_eff:.2f}")
        
        if 'stopping_reason' in df.columns:
            most_common_reason = df['stopping_reason'].mode().iloc[0] if not df['stopping_reason'].mode().empty else "N/A"
            print(f"Most common stopping reason: {most_common_reason}")

    # Best combinations
    print("\nBest Combinations:")
    best_overall = df.loc[df['best_val_acc'].idxmax()]
    best_epoch_info = f" (epoch {best_overall.get('best_epoch', 'N/A')})" if 'best_epoch' in df.columns else ""
    print(f"Best Overall: {best_overall['config_name']} ({best_overall['best_val_acc']:.4f}){best_epoch_info}")
    
    # Best for each APE type
    print("\nBest for each APE type:")
    for ape_type in df['ape_type'].unique():
        best_for_ape = df[df['ape_type'] == ape_type].loc[df[df['ape_type'] == ape_type]['best_val_acc'].idxmax()]
        epoch_info = f" (epoch {best_for_ape.get('best_epoch', 'N/A')})" if 'best_epoch' in df.columns else ""
        print(f"  {ape_type}: {best_for_ape['config_name']} ({best_for_ape['best_val_acc']:.4f}){epoch_info}")

def create_report(results_dir="vitarc_experiments"):
    """Create a comprehensive report"""
    
    results_file = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file)
    
    # Create markdown report
    report_path = os.path.join(results_dir, "experiment_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# ViTARC Positional Embedding Comparison Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Total experiments: {len(df)}\n")
        f.write(f"- Best validation accuracy: {df['best_val_acc'].max():.4f}\n")
        f.write(f"- Average validation accuracy: {df['best_val_acc'].mean():.4f}\n")
        f.write(f"- Total training time: {df['training_time'].sum():.1f} seconds\n\n")
        
        f.write("## Top 5 Configurations\n\n")
        top_configs = df.nlargest(5, 'best_val_acc')
        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            f.write(f"### {i}. {row['config_name']}\n")
            f.write(f"- **Validation Accuracy**: {row['best_val_acc']:.4f}\n")
            f.write(f"- **Description**: {row['description']}\n")
            f.write(f"- **APE Type**: {row['ape_type']}\n")
            f.write(f"- **RPE Type**: {row['rpe_type']}\n")
            f.write(f"- **Use OPE**: {row['use_OPE']}\n")
            f.write(f"- **Mixer Strategy**: {row['ape_mixer_strategy']}\n")
            f.write(f"- **Training Time**: {row['training_time']:.1f}s\n")
            f.write(f"- **Parameters**: {row['num_parameters']:,}\n")
            
            # Add early stopping information if available
            if 'best_epoch' in row:
                f.write(f"- **Best Epoch**: {row['best_epoch']}\n")
            if 'early_stopped' in row:
                status = "Yes" if row['early_stopped'] else "No"
                f.write(f"- **Early Stopped**: {status}\n")
            if 'convergence_efficiency' in row:
                f.write(f"- **Convergence Efficiency**: {row['convergence_efficiency']:.2f}\n")
            if 'stopping_reason' in row and row['early_stopped']:
                f.write(f"- **Stopping Reason**: {row['stopping_reason']}\n")
            f.write("\n")
        
        f.write("## Component Analysis\n\n")
        
        # APE analysis
        f.write("### APE Type Performance\n\n")
        ape_analysis = df.groupby('ape_type')['best_val_acc'].agg(['mean', 'max', 'count'])
        f.write("| APE Type | Mean Acc | Max Acc | Count |\n")
        f.write("|----------|----------|---------|-------|\n")
        for ape_type, data in ape_analysis.iterrows():
            f.write(f"| {ape_type} | {data['mean']:.4f} | {data['max']:.4f} | {data['count']} |\n")
        f.write("\n")
        
        # RPE analysis
        f.write("### RPE Type Performance\n\n")
        rpe_analysis = df.groupby('rpe_type')['best_val_acc'].agg(['mean', 'max', 'count'])
        f.write("| RPE Type | Mean Acc | Max Acc | Count |\n")
        f.write("|----------|----------|---------|-------|\n")
        for rpe_type, data in rpe_analysis.iterrows():
            f.write(f"| {rpe_type} | {data['mean']:.4f} | {data['max']:.4f} | {data['count']} |\n")
        f.write("\n")
        
        f.write("## Recommendations\n\n")
        best_config = df.loc[df['best_val_acc'].idxmax()]
        epoch_info = f" at epoch {best_config.get('best_epoch', 'N/A')}" if 'best_epoch' in df.columns else ""
        f.write(f"Based on the experimental results, the best configuration is **{best_config['config_name']}** ")
        f.write(f"with a validation accuracy of {best_config['best_val_acc']:.4f}{epoch_info}.\n\n")
        
        f.write("Key findings:\n")
        f.write(f"- **Best APE Type**: {df.groupby('ape_type')['best_val_acc'].mean().idxmax()}\n")
        f.write(f"- **Best RPE Type**: {df.groupby('rpe_type')['best_val_acc'].mean().idxmax()}\n")
        f.write(f"- **OPE Beneficial**: {df.groupby('use_OPE')['best_val_acc'].mean().idxmax()}\n")
        f.write(f"- **Best Mixer Strategy**: {df.groupby('ape_mixer_strategy')['best_val_acc'].mean().idxmax()}\n")
        
        # Add early stopping insights if available
        if 'early_stopped' in df.columns:
            early_stopped_pct = (df['early_stopped'].sum() / len(df)) * 100
            f.write(f"- **Early Stopping Rate**: {early_stopped_pct:.1f}% of experiments stopped early\n")
        
        if 'convergence_efficiency' in df.columns:
            avg_convergence = df['convergence_efficiency'].mean()
            f.write(f"- **Average Convergence Efficiency**: {avg_convergence:.2f} (models reach best performance at {avg_convergence*100:.1f}% of training)\n")
        
        if 'best_epoch' in df.columns:
            avg_best_epoch = df['best_epoch'].mean()
            f.write(f"- **Optimal Training Length**: Models typically reach best performance around epoch {avg_best_epoch:.0f}\n")
    
    print(f"Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Run ViTARC positional embedding comparison experiments')
    
    parser.add_argument('--action', choices=['run', 'analyze', 'report'], default='run',
                        help='Action to perform')
    parser.add_argument('--subset', type=str, help='Run only a subset of experiments')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--max-epochs', type=int, help='Maximum number of epochs per experiment')
    parser.add_argument('--results-dir', type=str, default='vitarc_experiments',
                        help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    if args.action == 'run':
        success = run_experiments(
            subset=args.subset,
            use_wandb=not args.no_wandb,
            max_epochs=args.max_epochs
        )
        
        if success:
            print("\nExperiments completed! Running analysis...")
            analyze_results(args.results_dir)
            create_report(args.results_dir)
    
    elif args.action == 'analyze':
        analyze_results(args.results_dir)
    
    elif args.action == 'report':
        create_report(args.results_dir)

if __name__ == "__main__":
    main() 