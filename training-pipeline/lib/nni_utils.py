import nni


def get_nni_trial_path():
    return f"experiment/{nni.get_experiment_id()}/{nni.get_sequence_id()}"
