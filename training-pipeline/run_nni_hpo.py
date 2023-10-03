import argparse
import json

from nni.experiment.config.training_services import LocalConfig
from nni.experiment import Experiment

if __name__ == "__main__":
    # CLI config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        help="Path locating the configuration file for the NNI experiment.",
                        type=str,
                        required=True,
                        default=False,)
    args = parser.parse_args()

    config = None
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    experiment = Experiment('local')
    # CUDA_VISIBLE_DEVICES=0 XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1
    experiment.config.trial_command = config["trial_command"]
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = config["search_space"]
    experiment.config.tuner.name = 'Anneal'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_experiment_duration = '72h'
    experiment.config.max_trial_number = 500
    experiment.config.trial_concurrency = 1
    experiment.config.training_service = LocalConfig(use_active_gpu=True)

    experiment.config.assessor.name = "Medianstop"
    experiment.config.assessor.class_args = {
        "start_step": config["assessor_start_step"],
        "optimize_mode": 'maximize'
    }

    experiment.run(32000)
    experiment.stop()
