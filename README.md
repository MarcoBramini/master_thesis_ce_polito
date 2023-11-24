# Designing an end-to-end Pipeline for Developing and Deploying IoT Solutions on Embedded Neuromorphic Platforms

*M.Sc. Thesis Project*  
Author: Marco Bramini  
Thesis Link: TBD

# How To Use
## Initialize Environment
### Create Conda Environment
This project was specifically thought to work on Linux OS, having Conda installed.
A fully comprehensive virtual environment can be installed on such OS using the following command:
```conda env create -f=requirements.txt -n myenv```

In case of different operating system, the user must manually build the virtual environment and install packages one by one:
```
conda create env -n myenv
conda activate myenv
conda install pip
pip install “rockpool[all]”
pip install nni
pip install tonic
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install ipykernel
```

### Unpack dataset
Use 7-Zip to unpack the multipart zip file.
The extraction must produce the following files: 

```
wisdm_watch_full_40.npz
wisdm_watch_full_40_encoded.npy
wisdm_watch_full_40_classes.json
```

The provided scripts assume these files are contained in a root folder named `data/`.

## Task Definition
The `task-definition` folder contains all the scripts needed for the Task Definition step.

### Generate tasks employing the DTW-based distance metric
1. Configure the global settings in the head of the script `task-definition/generate_tasks_dtw.py`.
2. Run the script with the command:
    ```
    cd task-definition
    python generate_tasks_dtw.py
    ```

### Generate tasks employing the KLD-based distance metric
1. Configure the global settings in the head of the script `task-definition/generate_tasks_kld.py`.
2. Run the script with the command:
    ```
    cd task-definition
    python generate_tasks_kld.py
    ```

### Plot the KDE for a specific combination of classes
1. Configure the global settings in the head of the script `task-definition/plot_kde.py`.
2. Run the script with the command:
    ```
    cd task-definition
    python plot_kde.py
    ```

## Model Generation
### Run Architecture Search
1. Configure the training pipeline editing the file `training-pipeline/run_xylo_rockpool_train.py`  

    In particular:
    - Select the model architecture editing the line 37, selecting one of the predefined architectures in the library:
        ```
        xylo_networks.get_ff_simple(...)
        xylo_networks.get_ff_deep(...)
        xylo_networks.get_ff_deep_res(...)
        xylo_networks.get_ff_deep_deep_res(...)
        xylo_networks.get_rec_simple(...)
        xylo_networks.get_rec_deep(...)
        ```
    - Edit the parameter `input_params["enabled_classes"]` at line 177, with the class labels associated to the task 7CB

2. Run NNI with the following commands:

    ```
    cd training-pipeline
    python run_nni_hpo.py --config_path nni_experiment_configs/xylo_as.json --port 32000
    ```
    It is possible to follow the process on the NNI GUI at `localhost:32000`

3. At the end of the process, the full list of experiments and model checkpoints will be stored in the folders `training-pipeline/experiments/{EXPERIMENT_ID}/{TRIAL_ID}`  

    The best performing model can be also retrieved with the script `training-pipeline/get_best_trial_id.py`, after setting the experiment id at line 4

### Run HPO
1. Configure the training pipeline editing the file `training-pipeline/run_(dynapse2|xylo)_rockpool_train.py`  

    In particular:
    - Edit the parameter `input_params["enabled_classes"]` at line 177, with the class labels associated to the selected task

2. Run NNI with the following commands:

    ```
    cd training-pipeline
    python run_nni_hpo.py --config_path nni_experiment_configs/(dynapse2|xylo)_hpo.json --port 32000
    ```

3. At the end of the process, the full list of experiments and model checkpoints will be stored in the folders `training-pipeline/experiments/{EXPERIMENT_ID}/{TRIAL_ID}`  

    The best performing model can be also retrieved with the script `training-pipeline/get_best_trial_id.py`, after setting the experiment id at line 4

### Run Extended Training
1. Configure the training pipeline editing the file `training-pipeline/run_(dynapse2|xylo)_rockpool_train.py`  

    In particular:
    - Edit the parameter `input_params["enabled_classes"]` at line 177, with the class labels associated to the selected task

2. Directly run the training pipeline with the following command: 
    ```
    cd training-pipeline
    PYTHONPATH="lib" python run_(dynapse2|xylo)_rockpool_train.py
    ```
3. At the end of the process, the folder `best_model/` will contain the checkpoint of the trained model

## Hardware Deployment
### Run Quantization Tuning
This process needs for a trained model. The tuning parameters will be automatically applied in the model checkpoint.

1. Configure the NNI Quantization Tuning experiment configuration by editing the file `training-pipeline/nni_experiment_configs/(dynapse2|xylo)_tuning_(task).json`  

    In particular:
    - Change the model path at line 2.

2. Run NNI with the following commands:

    ```
    cd training-pipeline
    python run_nni_hpo.py --config_path nni_experiment_configs/(dynapse2|xylo)_tuning_(task).json --port 32000
    ```

### Hardware Configuration Generation
Refer to the Python Notebooks in `dynapse2-deploy/`and `xylo-deploy/` for the generation of the hardware configuration.
