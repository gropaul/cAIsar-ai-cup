import numpy as np

config_8c58c462 = {
  "session_param_id": 1241,
  "bg_train_config": {
    "data_subset_name": "gait_training",
    "normalization_settings": {
      "normalization_function": "min_max_symmetrical",
      "normalization_mode": "by_all"
    },
    "data_columns": [
      "LAV",
      "LAX",
      "LAY",
      "LAZ",
      "LRV",
      "LRX",
      "LRY",
      "LRZ",
      "RAV",
      "RAX",
      "RAY",
      "RAZ",
      "RRV",
      "RRX",
      "RRY",
      "RRZ"
    ],
    "label_columns": [
      "LFA",
      "RFA"
    ],
    "batch_size": 64,
    "shuffle": True,
    "dtype": {
      "LAV": np.float64,
      "LAX": np.float64,
      "LAY": np.float64,
      "LAZ": np.float64,
      "LRV": np.float64,
      "LRX": np.float64,
      "LRY": np.float64,
      "LRZ": np.float64,
      "RAV": np.float64,
      "RAX": np.float64,
      "RAY": np.float64,
      "RAZ": np.float64,
      "RRV": np.float64,
      "RRX": np.float64,
      "RRY": np.float64,
      "RRZ": np.float64,
      "LFA": np.int8,
      "RFA": np.int8
    },
    "verbose": True,
    "length": 1024,
    "validation_generator": False,
    "train_test_split": 1.0,
    "sample_freq": 512,
    "sample_offset": 0,
    "caching": True,
    "padding": True,
    "pad_last_batch": True,
    "meta": True
  },
  "bg_evaluation_config": {
    "data_subset_name": "gait_evaluation",
    "normalization_settings": {
      "normalization_function": "min_max_symmetrical",
      "normalization_mode": "by_all"
    },
    "data_columns": [
      "LAV",
      "LAX",
      "LAY",
      "LAZ",
      "LRV",
      "LRX",
      "LRY",
      "LRZ",
      "RAV",
      "RAX",
      "RAY",
      "RAZ",
      "RRV",
      "RRX",
      "RRY",
      "RRZ"
    ],
    "label_columns": [
      "LFA",
      "RFA"
    ],
    "batch_size": 64,
    "shuffle": False,
    "dtype": {
      "LAV": np.float64,
      "LAX": np.float64,
      "LAY": np.float64,
      "LAZ": np.float64,
      "LRV": np.float64,
      "LRX": np.float64,
      "LRY": np.float64,
      "LRZ": np.float64,
      "RAV": np.float64,
      "RAX": np.float64,
      "RAY": np.float64,
      "RAZ": np.float64,
      "RRV": np.float64,
      "RRX": np.float64,
      "RRY": np.float64,
      "RRZ": np.float64,
      "LFA": np.int8,
      "RFA": np.int8
    },
    "verbose": True,
    "length": 1024,
    "validation_generator": True,
    "train_test_split": 0.0,
    "sample_freq": 512,
    "sample_offset": 0,
    "caching": True,
    "padding": True,
    "pad_last_batch": True,
    "meta": True
  },
  "data_augmenter_config": {
    "seed": None,
    "drift_max": 0.07964866515640778,
    "drift_points": 72,
    "drift_kind": "multiplicative",
    "drift_prob": 0.47839777566007474,
    "noise_scale": 0.044058087903574225,
    "noise_prob": 0.3974174382427959,
    "convolve_window_type": "bohman",
    "convolve_window_size": 5,
    "convolve_prob": 0.9234514530170541,
    "dropout_percentage": 0.3989302210801908,
    "dropout_prob": 0.7082288529739441,
    "dropout_size": 8,
    "dropout_fill": "bfill",
    "time_warp_changes": 47,
    "time_warp_max": 2.406625020809407,
    "time_warp_prob": 0.6304581978724473
  },
  "model_config": {
    "sequence_length": 1024,
    "channels": 16,
    "base_architecture": "U-net",
    "additional_architectures": [
      "Dense",
      #"Inception",
      "Residual"
    ],
    "base_filter_count": 8,
    "backbone_length": 4,
    "concat_axis": 2,
    "class_count": 2,
    "kernel_size": {
      -1: 8
    },
    "pool_size": {
      -1: 4,
      "0": {
        "0": 4
      },
      "1": {
        "0": 4
      },
      "2": {
        "0": 4
      },
      "3": {
        "0": 4
      }
    },
    "dropout": {
      -1: 0.4000026933775057
    },
    "strides": {
      -1: 1,
      "0": {
        "0": {
          "0": 1
        },
        "1": {
          "0": 1
        },
        "2": {
          "0": 1
        },
        "3": {
          "0": 1
        }
      },
      "1": {
        "0": {
          "0": 1
        },
        "1": {
          "0": 1
        },
        "2": {
          "0": 1
        }
      },
      "2": {
        "0": {
          "0": 1
        },
        "1": {
          "0": 1
        }
      },
      "3": {
        "0": {
          "0": 1
        }
      }
    },
    "n_fold_convolutions": 1,
    "attention_kernel": {
      -1: 3
    },
    "attention_intermediate": {
      -1: 0.5305714484766109
    },
    "inception_kernel_size": {
      -1: 6
    },
    "meta_length": 14,
    "meta_dropout": 0.5542161639996841,
    "post_dense_meta_dropout": {
      -1: 0.0
    }
  },
  "training_config": {
    "epochs": 40,
    "loss": "tversky_loss",
    "tversky_beta": 0.8,
    "learning_rate": 0.01,
    "optimizer": "Adam",
    "lr_epoch_per_step": 5,
    "lr_increase_per_step": 0.1,
    "lr_number_of_steps": 4,
    "lr_start": 0.1
  }
}