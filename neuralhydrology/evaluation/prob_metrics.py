import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Union

def run_probabilistic_evaluation(obs: np.ndarray, sim_samples: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """Performs a probabilistic evaluation of predictive samples against observations.

    This function replicates the reliability and resolution analysis from Klotz et al. (2020).
    - Reliability is assessed using a Probability Integral Transform (PIT) plot.
    - Resolution (sharpness) is assessed using statistics of the predictive distribution.

    Args:
        obs (np.ndarray): 1D array of observed values (e.g., streamflow).
        sim_samples (np.ndarray): 2D array of corresponding predictive samples, with
                                  shape (n_timesteps, n_samples).

    Returns:
        Dict[str, Union[float, np.ndarray]]: A dictionary containing the calculated
                                             resolution metrics and the PIT values.
    """
    # --- Input Validation ---
    if obs.ndim != 1:
        raise ValueError("Observations (`obs`) must be a 1D array.")
    if sim_samples.ndim != 2:
        raise ValueError("Simulated samples (`sim_samples`) must be a 2D array.")
    if len(obs) != sim_samples.shape[0]:
        raise ValueError("Observations and simulations must have the same number of timesteps.")
        
    print(f"Performing probabilistic evaluation on {len(obs)} timesteps with {sim_samples.shape[1]} samples each.")

    # --- 1. Reliability Analysis (PIT) ---
    print("\nCalculating reliability...")
    
    # Calculate PIT values: fraction of samples <= observation for each timestep
    pit_values = np.mean(sim_samples <= obs[:, np.newaxis], axis=1)

    # Prepare data for the probability plot
    thresholds = np.arange(0.1, 1.1, 0.1)
    observed_quantiles = [np.mean(pit_values <= q) for q in thresholds]
    
    # --- 2. Resolution (Sharpness) Analysis ---
    print("Calculating resolution (sharpness)...")
    
    # Calculate sharpness metrics for each timestep's prediction
    std_devs = np.std(sim_samples, axis=1)
    iqr = np.percentile(sim_samples, 75, axis=1) - np.percentile(sim_samples, 25, axis=1)
    interdecile_range = np.percentile(sim_samples, 90, axis=1) - np.percentile(sim_samples, 10, axis=1)
    
    # Average the metrics over all timesteps
    resolution_metrics = {
        'mean_std_dev': np.mean(std_devs),
        'mean_interquartile_range': np.mean(iqr),
        'mean_interdecile_range': np.mean(interdecile_range)
    }

    # --- 3. Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # a) Probability Plot
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k:', label="Perfectly reliable")
    ax.plot(thresholds, observed_quantiles, 'o-', label="Model")
    ax.set_xlabel("Theoretical Quantiles (Uniform)")
    ax.set_ylabel("Observed Quantiles")
    ax.set_title("a) Reliability Plot")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # b) Deviation Plot
    ax = axes[1]
    deviations = observed_quantiles - thresholds
    ax.bar(thresholds, deviations, width=0.08)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Deviation")
    ax.set_title("b) Deviation Plot")
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()

    # --- 4. Print results ---
    print("\nResolution Metrics (lower is better):")
    for key, val in resolution_metrics.items():
        print(f"  - {key}: {val:.4f}")
        
    # Add PIT values to the returned dictionary for further analysis if needed
    resolution_metrics['pit_values'] = pit_values
        
    return resolution_metrics