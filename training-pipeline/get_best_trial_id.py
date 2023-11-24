import os
import json

EXPERIMENT_ID = "du6ejcys"

if __name__ == "__main__":
    experiment_path = f"experiment/{EXPERIMENT_ID}"

    best_accuracy = 0
    best_trial_id = -1
    for trial_id in os.listdir(experiment_path):
        trial_metadata_path = f"{experiment_path}/{trial_id}/best_model_metadata.json"
        try:
            with open(trial_metadata_path) as f:
                trial_metadata = json.load(f)
                if trial_metadata["val_accuracy"] > best_accuracy:
                    best_trial_id = trial_id
                    best_accuracy = trial_metadata["val_accuracy"]
        except:
            continue

    print(best_trial_id, best_accuracy)
