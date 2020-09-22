# ISVC2020: Overscan Detection in Digitized Analog Films by Precise Sprocket Hole Segmentation

Requirements:

    pip install tqdm, torch, torchvision, tensorboard, tensorboardX, mlflow, sklearn, matplotlib 

Structure:

    ./configs: includes the configuration files for the executed experiments (json format)
    ./templates: asdfa sdfa sd fasd fasd 
    ./scripts: includes some helper scripts
    ./dataset_generator: includes functions to create our dataset
    ./pre_trained_models: includes pre_trained models
    models.py: includes all model definitions
    metrics.py: includes all metrics used for this evaluation (dice, IoU, ...)
    dataset.py: includes the dataloader and Dataset interface.
    utils.py: includes some helper functions
    segmentation_train.py: includes functionality to start training process
    segmentation_inference.py: includes the functionaltiy to start inference mode
    requirements.txt: includes a list of all necessary libraries

Dataset: 

    prepare dataset

Pre-Trained models:


Run Training: 

    to start the training process the following commands are needed:
    python segmentation_train.py -m <SELECT_MODE> [optional: -c <PATH_TO_CONFIG>]
    
    e.g. 
    --> start training with single configuration file (.json see ./config folder)
    python segmentation_train.py -m train_single -c ./config/debug/train_exp_debug.json

    --> start training on multiple configuration files (.json see ./config folder)
    python segmentation_train.py -m train_multiple 
    Hint: change configuration path in method trainMultipleExp(...)
    
    --> start evaluation of single experiment (experiment folder includes experiment_notes.json, best_model.pth and loss_history.log)
    python segmentation_train.py -m train_single -c ./config/debug/train_exp_debug.json
    
    --> start evaluation on multiple experiments 
    python segmentation_train.py -m test_multiple 
    Hint: change results path in method runMultipleTests(...)


Run Inference:

    to run the a pre-trained model in inference mode the following commands are needed:
    python segmentation_inference.py
    
    e.g. 
    --> start training with single configuration file (.json see ./config folder)
    python segmentation_train.py -m train_single -c ./config/debug/train_exp_debug.json

    --> start training on multiple configuration files (.json see ./config folder)
    python segmentation_train.py -m train_multiple 
    Hint: change configuration path in method trainMultipleExp(...)


