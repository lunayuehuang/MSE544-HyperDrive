<font color="red"># TO DO: 
- Include screenshots
- Include detail/descriptions of what is happening
- Include other options not mentioned in tutorial (i.e. sampling types)
- Properly format document 
- Proof read & spell check
</font>

# MSE544-HyperDrive Experiment
 
## Repository Background
The framework presented in this work introduces the crystal graph convolution neural networks (CGCNN), which are designed to represent periodic crystal systems and predict material properties at DFT level accuracy and propose chemical insight. Read more about this study [here](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.145301).  
## Dataset Introduction
A collection of 3,210 .cif crystal structures have been extracted from the "materials project" website and consolidated into Azure data storage. <font color="red">be more detailed </font> 

-----------------------------------
## Instructions
### Part I: Set up the repository
1. Make a directory 
    ``` 
    mkdir MSE544-Hyperdrive
    ```
2. Move into the new directory
    ```
    cd MSE544-Hyperdrive
    ```
3. Clone the repository we will be using
    ```
    git clone https://github.com/txie-93/cgcnn.git
    ```
4. Move into the "cgcnn" directory 
    ```
    cd cgcnn
    ````
5. Download the .yml file from Canvas and place it in the "cgcnn" directory. The .yml (sometimes seen as .yaml) file is a special file typically used for configuring environments/settings for programs. Files with this extension are intended to be human-readable.
    FUN FACT: YAML initially stood for, *Yet Another Markdown Language*
### Part II: Build the Notebook
1. Make a jupyter notebook called "hyperdrive_experiment" - make sure this notebook is in the same directory as the "main.py" python script
2. Insert a cell with the following imports
    ```
    from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset, Run
    import azureml
    import os, tempfile, tarfile
    from azureml.train.hyperdrive import BayesianParameterSampling
    from azureml.train.hyperdrive import normal, uniform, choice
    from azureml.core.run import Run
    from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal
    ```
3. Initialize a workspace in the next cell (be sure to enter the appropriate information)
    ```
    subscription_id = <INSERT HERE>
    resource_group  = <INSERT HERE>
    workspace_name  = <INSERT HERE>
    ws = Workspace(subscription_id, resource_group, workspace_name)
    experiment = Experiment(workspace=ws, name='hyperdrive_experiment')
    ```
4. Create a "dataset" variable to represent the .cif files we will use for training - <font color="red">THIS NEEDS TO BE UPDATED TO WORK WITH A KEY</font>
    ```
    dataset = Dataset.get_by_name(ws, name='materials_project_3207_unzipped')
    ```
5. Set an environment variable using the .yml file
    ```
    cgcnn_env = Environment.from_conda_specification(name='cgcnn_env', file_path='cgcnn_env.yml')
    ```
6. Configurate the base training session
    Here we are configuring our experiment, as we have done in previous tutorials.
    - *source_directory:* indicates the (working) directory our scripts can be found
    - *script:* defines the python script we want to run
    - *compute_target:* tells Azure where we want to run this experiment
    - *environment:* initiates the predefined environment needed to succesfully run this experiment
    - *arguments:* allows us to define some constant parameters that the experiment should use (i.e. ratio of data allocatted to the test, validation, and training set). Notice we also input our dataset here, which we have mounted previously
    ```
    config = ScriptRunConfig(source_directory='./',   
                             script='main-hyper.py',       
                             compute_target='<INSERT HERE>', 
                             environment=cgcnn_env,
                             arguments=[
                                '--epochs', 5,
                                '--train-ratio', 0.6,
                                '--val-ratio', 0.2,
                                '--test-ratio', 0.2, 
                                 dataset.as_named_input('input').as_mount()]                   
                             )
    ```
7. Define the parameters you are interested in sampling

    In setting up our search space, we have the option of defining discrete or continous hyperparameter spaces where the former is initiated by "choice" and the latter can be requested via "uniform" (amongst others)
    ```
    param_sampling = BayesianParameterSampling( {
        "batch-size": choice(16, 32, 64),
        "learning-rate": uniform(0.05, 0.1),
        "optim": choice("SGD", "Adam")
    }
    )
    ```
    There are three different methods in which the hyperparameter space can be sampled: 
    i. *Random sampling*: hyperparameters are randomly selected from the defined search space 
    ii. *Grid sampling*: hyperparameters are selected such that all possible combinations are explored during experimentation (computationally expensive)
    iii. *Bayesian sampling*: hyperparameters are selected based on the outcomes of previous experiments; each subsequent run should be an improvement over the previous
    (See reference I. for addition details)
8. Configure the hyperdrive experiment
    ```
    hd_config = HyperDriveConfig(run_config=config,
                                 hyperparameter_sampling=param_sampling,
                                 primary_metric_name="MAE",
                                 primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                 max_total_runs=8,
                                 max_concurrent_runs=4)
    ```
9. Finally, run the experiment and monitor the progress at the printed url
    ```
    run = experiment.submit(hd_config)
    aml_url = run.get_portal_url()
    print(aml_url)
    ```

### Part III: Running the Experiment and Navigating Azure
1. When you follow the url printed in step 9 of part II, you should find a page that looks something like this:
<img src="./images/follow_url_notes.png" style="height: 90%; width: 90%;"/>
    A. Pathway to the experiment we are running 
    B. Name of the current experiment - this is easily edited to something more meaningful by selecting the pencil symbol 
    C. Tab showing the various runs that will be submitted during the experiment
2. Select the "child runs" tab to view the following page:
<img src="./images/child_runs_notes.png" style="height: 90%; width: 90%;"/>
    A. Lists the subsequent runs within my experiment and provides relevant information such as: name of the run, status (pending, queued, complete), mean absolute error (MAE), duration of the run, batch size, time submitted. Notice the small arrow next to MAE, which indicates that I have sorted my runs based on the resulting MAE value. 
    B. Visualization of the MAE for each run as they progressed
    C. Chart correlating the hyperparameters selected for each run and the calculated MAE 




# References
I. [Hyperparameter tuning models using Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#define-search-space)