from typing import Tuple
from tsaug import AddNoise, Dropout, TimeWarp,Reverse,Resize,Convolve,Drift
from utils.util_functions import printc
from data_generator.data_augmenter_params import DataAugmenterParams


class DataAugmenter:

    SEED = 42

    v1 = (
        AddNoise(loc=0.0, scale=(0.0,0.4), distr='gaussian', kind='additive', per_channel=True, normalize=True, prob=1.0, seed=SEED)
        + Dropout(p=(0,0.4), size=[i for i in range(1, 5)], per_channel=True, prob=1, seed=SEED)
        + TimeWarp(n_speed_change=3, max_speed_ratio=(1.01, 1.05), repeats=1, prob=1, seed=SEED)
        + Reverse(prob = 0.2)
    )


    default = None

    def send_message(message: str, **kwargs) -> None:
        printc(source='[DataAugmenter]', message=message, **kwargs)

    @staticmethod
    def get_custom(seed: int = 42, 
        noise_scale: float = 0, noise_prob: float = 0,
        drift_max: float = 0, drift_points: int = 0, drift_kind : str = 'additive', drift_prob: float = 0.0, 
        convolve_window_type: str = None, convolve_window_size: int = 1, convolve_prob: float = 0.0,
        dropout_percentage: float = 0, dropout_size: int = 0, dropout_fill: str = "mean", dropout_prob: float = 0.0,
        time_warp_changes: int = 3, time_warp_max: float = 3.0, time_warp_prob = 0.0
    ):
        custom = AddNoise(loc=0.0, scale=(0.0,0.00000001), distr='gaussian', kind='additive', per_channel=False, normalize=False, prob=1.0, seed=seed) 
        
        if noise_prob != 0.0:
            custom += AddNoise(loc=0, scale=noise_scale, distr='gaussian', kind='additive', per_channel=False, normalize=False, prob=noise_prob, seed=seed)

        if time_warp_prob != 0.0:
            custom += TimeWarp(n_speed_change=time_warp_changes, max_speed_ratio=time_warp_max, prob=time_warp_prob, seed=seed)

        if drift_max != 0.0:
             custom += (Drift(max_drift=drift_max, n_drift_points=drift_points, seed=seed, per_channel=False, kind=drift_kind, prob = drift_prob))


        if dropout_percentage != 0.0:
            custom +=  Dropout(p=dropout_percentage, size=dropout_size, per_channel=False,fill=dropout_fill, prob=dropout_prob, seed=seed)

        if convolve_window_type != None:
            DataAugmenter.send_message(f"Added a Convolve Augmenter with window_type={convolve_window_type} and window_size={convolve_window_size}")
            custom += Convolve(window=convolve_window_type,size=convolve_window_size, prob=convolve_prob, seed=seed)

        return custom
