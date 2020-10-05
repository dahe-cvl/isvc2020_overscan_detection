# ISVC2020: Overscan Detection in Digitized Analog Films by Precise Sprocket Hole Segmentation

Paper title: Overscan Detection in Digitized Analog Films by Precise Sprocket Hole Segmentation
Conference: ISVC 2020 - 15th International Symposium on Visual Computing


Requirements:

    pip install tqdm, torch, torchvision, tensorboard, tensorboardX, mlflow, sklearn, matplotlib 

Structure:

    ./configs: includes the configuration files for the executed experiments (json format)
    ./templates: includes the film reel templates (16mm vs 9.5)
    ./scripts: includes some helper scripts
    ./DatasetGenerator: includes functions to create our dataset
    ./pre_trained_models: includes pre_trained models
    models.py: includes all model definitions
    metrics.py: includes all metrics used for this evaluation (dice, IoU, ...)
    dataset.py: includes the dataloader and Dataset interface.
    utils.py: includes some helper functions
    segmentation_train.py: includes functionality to start training process
    segmentation_inference.py: includes the functionaltiy to start inference mode
    evaluate_final_results.py: is used to calculate final metrics scores for specified dataset
    requirements.txt: includes a list of all necessary libraries

Dataset: 

    Training and Validation set: 
    This dataset is based on the benchmark dataset MS COCO. [https://cocodataset.org/#home]
    
    Test set:
    Because of copyright issues the image samples of the test set cannot be published. However a list
    as well as the ground truth masks are published in zenodo.
    Ephemeral Films Project: [http://efilms.ushmm.org/]
    
    Download: [https://doi.org/10.5281/zenodo.4065499]
    
    Hint: Download all archive parts and extract it into one defined archive.
    
Pre-Trained models:
    
    The pre-trained models used in this investigation are published in zenodo. This model can be used for reproducing
    the results published in the paper "Overscan Detection in Digitized Analog Films by Precise Sprocket Hole Segmentation"
    at the conference ISVC2020.
    
    Download: [https://doi.org/10.5281/zenodo.4065499]


Run Training: 

    to start the training process the following commands are needed:
    python segmentation_train.py -m <SELECT_MODE> [optional: -c <PATH_TO_CONFIG>]
    
    e.g. 
    --> start training with single configuration file (.json see ./config folder)
    python segmentation_train.py -m train_single -c ./config/debug/train_exp_debug.json

    --> start training on multiple configuration files (.json see ./config folder)
    python segmentation_train.py -m train_multiple 
    Hint: change configuration path in method trainMultipleExp(...)
        
    --> start evaluation on multiple experiments 
    python segmentation_train.py -m test_multiple 
    Hint: change results path in method runMultipleTests(...)


Run Inference:

    to run the a pre-trained model in inference mode the following commands are needed:
    python segmentation_inference.py
    
