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
        
        # Access the full model outputs from args[0]
        # This dictionary contains y_hat_1D, y_hat_1h for the *entire sequence*
        if args:
            full_model_outputs = args[0]
        else:
            # Fallback or raise an error if full_model_outputs is expected but not provided
            # This might happen if BaseLoss doesn't pass it or if the setup is incorrect.
            # In a typical NeuralHydrology setup, args[0] should be present.
            raise RuntimeError("Full model outputs not found in *args. TiedFrequencyMSERegularization requires it.")


        for idx, freq in enumerate(self._frequencies):
            if idx == 0:
                continue
            
            frequency_factor = int(get_frequency_factor(self._frequencies[idx - 1], freq))
            
            # Use full_model_outputs for the high-frequency prediction
            high_res_y_hat = full_model_outputs[f'y_hat_{freq}']
            
            # Aggregate the high-resolution prediction to the lower frequency
            # Ensure the reshaping logic aligns with your data dimensions (batch, sequence_length, features)
            mean_high_res_y_hat = high_res_y_hat.view(high_res_y_hat.shape[0], 
                                                        high_res_y_hat.shape[1] // frequency_factor,
                                                        frequency_factor, -1).mean(dim=2)
            
            # Use full_model_outputs for the lower-frequency prediction
            low_res_y_hat = full_model_outputs[f'y_hat_{self._frequencies[idx - 1]}']
            
            # Take the last relevant timesteps for comparison
            # Ensure slicing is correct based on the aggregated shape
            lower_freq_pred_cropped = low_res_y_hat[:, -mean_high_res_y_hat.shape[1]:]
            
            # Calculate the mean squared error
            loss = loss + torch.mean((lower_freq_pred_cropped - mean_high_res_y_hat)**2)

        return loss

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
