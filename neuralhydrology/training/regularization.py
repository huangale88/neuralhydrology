from typing import Dict

import pandas as pd
import torch

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Base class for regularization terms.

    Regularization terms subclass this class by implementing the `forward` method.

    Parameters
    ----------
    cfg: Config
        The run configuration.
    name: str
        The name of the regularization term.
    weight: float, optional.
        The weight of the regularization term. Default: 1.
    """

    def __init__(self, cfg: Config, name: str, weight: float = 1.0):
        super(BaseRegularization, self).__init__()
        self.cfg = cfg
        self.name = name
        self.weight = weight

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the regularization term.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary of predicted variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary of ground truth variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        other_model_data : Dict[str, torch.Tensor]
            Dictionary of all remaining keys-value pairs in the prediction dictionary that are not directly linked to 
            the model predictions but can be useful for regularization purposes, e.g. network internals, weights etc.
            
        Returns
        -------
        torch.Tensor
            The regularization value.
        """
        raise NotImplementedError

class TiedFrequencyMSERegularization(BaseRegularization):
    """Regularization that penalizes inconsistent predictions across frequencies.

    This regularization can only be used if at least two frequencies are predicted. For each pair of adjacent
    frequencies f and f', where f is a higher frequency than f', it aggregates the f-predictions to f' and calculates
    the mean squared deviation between f' and aggregated f.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    weight: float, optional.
        Weight of the regularization term. Default: 1.

    Raises
    ------
    ValueError
        If the run configuration only predicts one frequency.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(TiedFrequencyMSERegularization, self).__init__(cfg, name='tie_frequencies', weight=weight)
        self._frequencies = sort_frequencies(
            [f for f in cfg.use_frequencies if cfg.predict_last_n[f] > 0 and f not in cfg.no_loss_frequencies])

        if len(self._frequencies) < 2:
            raise ValueError("TiedFrequencyMSERegularization needs at least two frequencies.")

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                *args) -> torch.Tensor:
        """Calculate the sum of mean squared deviations between adjacent predicted frequencies.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary containing ``y_hat_{frequency}`` for each frequency.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary continaing ``y_{frequency}`` for each frequency.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations for each pair of adjacent frequencies.
        """

        loss = 0
        for idx, freq in enumerate(self._frequencies):
            if idx == 0:
                continue
            frequency_factor = int(get_frequency_factor(self._frequencies[idx - 1], freq))
            freq_pred = prediction[f'y_hat_{freq}']
            mean_freq_pred = freq_pred.view(freq_pred.shape[0], freq_pred.shape[1] // frequency_factor,
                                            frequency_factor, -1).mean(dim=2)
            lower_freq_pred = prediction[f'y_hat_{self._frequencies[idx - 1]}'][:, -mean_freq_pred.shape[1]:]
            loss = loss + torch.mean((lower_freq_pred - mean_freq_pred)**2)

        return loss

class TiedFrequencyMSERegularizationCMAL(BaseRegularization):
    """Regularization that penalizes inconsistent predictions across frequencies.

    This regularization can only be used if at least two frequencies are predicted. For each pair of adjacent
    frequencies f and f', where f is a higher frequency than f', it aggregates the f-predictions to f' and calculates
    the mean squared deviation between f' and aggregated f.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    weight: float, optional.
        Weight of the regularization term. Default: 1.

    Raises
    ------
    ValueError
        If the run configuration only predicts one frequency.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(TiedFrequencyMSERegularizationCMAL, self).__init__(cfg, name='tie_frequencies_cmal', weight=weight)
        self._frequencies = sort_frequencies(
            [f for f in cfg.use_frequencies if cfg.predict_last_n[f] > 0 and f not in cfg.no_loss_frequencies])

        if len(self._frequencies) < 2:
            raise ValueError("TiedFrequencyMSERegularization needs at least two frequencies.")
        
    def _calculate_cmal_mean(self, mu: torch.Tensor, b: torch.Tensor, tau: torch.Tensor, pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculates the mean of the CMAL mixture distribution."""
        tau_safe = torch.clamp(tau, eps, 1.0 - eps)
        component_means = mu + b * (1.0 - 2.0 * tau_safe) / (tau_safe * (1.0 - tau_safe))
        pi_safe = pi / (pi.sum(dim=-1, keepdim=True) + eps)
        mean_mixture = torch.sum(pi_safe * component_means, dim=-1, keepdim=True)
        return mean_mixture

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                *args) -> torch.Tensor:
        """Calculate the sum of mean squared deviations between adjacent predicted frequencies.

        This version is adapted for CMAL heads. It checks if a point-prediction 'y_hat'
        exists. If not (e.g., during evaluation), it calculates the mean of the CMAL
        distribution from its raw parameters before comparing frequencies.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary containing model predictions. During training, this will contain
            y_hat_[freq]. During evaluation, it will contain mu_[freq], b_[freq], etc.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary containing ground truth values. Not used in this regularization.
        *args :
            Additional arguments. Expects the full, un-sliced model output dictionary
            as the first element.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations for each pair of adjacent frequencies.
        """
        if not args:
            raise RuntimeError("Full model outputs not found in *args. TiedFrequencyMSERegularization requires it.")
        
        full_model_outputs = args[0]
        total_loss = 0.0

        # Frequencies are sorted from coarse to fine (e.g., 4W-MON, 2W-MON)
        for idx, freq in enumerate(self._frequencies):
            # We start from the second frequency to compare it with the previous (coarser) one.
            if idx > 0:
                coarse_freq = self._frequencies[idx - 1]
                fine_freq = freq
                
                coarse_y_hat_key = f'y_hat_{coarse_freq}'
                fine_y_hat_key = f'y_hat_{fine_freq}'

                # --- THIS IS THE CRITICAL LOGIC ---

                # 1. Get the coarse-resolution y_hat (point prediction)
                if coarse_y_hat_key in full_model_outputs:
                    coarse_y_hat = full_model_outputs[coarse_y_hat_key]
                else: # If it doesn't exist, we are in eval mode. Calculate it from CMAL params.
                    coarse_y_hat = self._calculate_cmal_mean(
                        full_model_outputs[f'mu_{coarse_freq}'],
                        full_model_outputs[f'b_{coarse_freq}'],
                        full_model_outputs[f'tau_{coarse_freq}'],
                        full_model_outputs[f'pi_{coarse_freq}']
                    )

                # 2. Get the fine-resolution y_hat (point prediction)
                if fine_y_hat_key in full_model_outputs:
                    fine_y_hat = full_model_outputs[fine_y_hat_key]
                else: # If it doesn't exist, calculate it from CMAL params.
                    fine_y_hat = self._calculate_cmal_mean(
                        full_model_outputs[f'mu_{fine_freq}'],
                        full_model_outputs[f'b_{fine_freq}'],
                        full_model_outputs[f'tau_{fine_freq}'],
                        full_model_outputs[f'pi_{fine_freq}']
                    )
                
                # 3. Aggregate the fine-resolution prediction to the coarse resolution
                frequency_factor = int(get_frequency_factor(coarse_freq, fine_freq))
                
                # Reshape and take the mean to aggregate
                aggregated_fine_y_hat = fine_y_hat.view(
                    fine_y_hat.shape[0], # batch size
                    fine_y_hat.shape[1] // frequency_factor, # new sequence length
                    frequency_factor, # number of fine steps per coarse step
                    -1 # feature dimension
                ).mean(dim=2)
                
                # 4. Calculate the MSE loss between the two
                # We need to make sure they have the same sequence length for comparison
                if coarse_y_hat.shape[1] != aggregated_fine_y_hat.shape[1]:
                    # This can happen due to slicing. We compare the overlapping part at the end.
                    num_timesteps = min(coarse_y_hat.shape[1], aggregated_fine_y_hat.shape[1])
                    coarse_y_hat_cropped = coarse_y_hat[:, -num_timesteps:]
                    aggregated_fine_y_hat_cropped = aggregated_fine_y_hat[:, -num_timesteps:]
                else:
                    coarse_y_hat_cropped = coarse_y_hat
                    aggregated_fine_y_hat_cropped = aggregated_fine_y_hat
                
                # Using a built-in MSE loss function is cleaner
                loss_func = torch.nn.MSELoss()
                total_loss += loss_func(aggregated_fine_y_hat_cropped, coarse_y_hat_cropped)

        return total_loss

class ForecastOverlapMSERegularization(BaseRegularization):
    """Squared error regularization for penalizing differences between hindcast and forecast models.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config, weight: float = 1.0):
        super(ForecastOverlapMSERegularization, self).__init__(cfg, name='forecast_overlap', weight=weight)

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_output: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Calculate the squared difference between hindcast and forecast model during overlap.

        Does not work with multi-frequency models.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Not used.
        ground_truth : Dict[str, torch.Tensor]
            Not used.
        other_model_output : Dict[str, Dict[str, torch.Tensor]]
            Dictionary containing ``y_forecast_overlap`` and ``y_hindcast_overlap``, which are
            both dictionaries containing keys to relevant model outputs.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations between overlapping portions of hindcast and forecast models.

        Raises
        ------
        ValueError if y_hindcast_overlap or y_forecast_overlap is not present in model output.
        """
        loss = 0
        if 'y_hindcast_overlap' not in other_model_output or not other_model_output['y_hindcast_overlap']:
            raise ValueError('y_hindcast_overlap is not present in the model output.')
        if 'y_forecast_overlap' not in other_model_output or not other_model_output['y_forecast_overlap']:
            raise ValueError('y_forecast_overlap is not present in the model output.')
        for key in other_model_output['y_hindcast_overlap']:
            hindcast = other_model_output['y_hindcast_overlap'][key]
            forecast = other_model_output['y_forecast_overlap'][key]
            loss += torch.mean((hindcast - forecast)**2)
        return loss
