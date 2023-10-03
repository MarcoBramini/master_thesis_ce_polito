import os
import json

experiment_id = "du6ejcys"

if __name__ == "__main__":
    # List all folder in path/experimentID
    # Read metadata within them and get accuracy
    experiment_path = f"experiment/{experiment_id}"

    best_accuracy = 0
    best_trial_id = -1
    for trial_id in os.listdir(experiment_path):
        trial_metadata_path = f"{experiment_path}/{trial_id}/best_model_metadata.json"
        try:
            with open(trial_metadata_path) as f:
                trial_metadata = json.load(f)
                if trial_metadata["val_accuracy"] > best_accuracy and trial_metadata["params"]["n_population"] == 64:
                    best_trial_id = trial_id
                    best_accuracy = trial_metadata["val_accuracy"]
        except:
            continue

    print(best_trial_id, best_accuracy)
