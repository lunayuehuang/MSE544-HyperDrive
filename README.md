<font color="red"># TO DO: 
- Include screenshots
- Include detail/descriptions of what is happening
- Include other options not mentioned in tutorial (i.e. sampling types)
- Properly format document 
</font>
# MSE544-HyperDrive
Tutorial for MSE 544 Azure ML HyperDrive  

# Dataset Introduction

# Instructions
## Part I: Building the Tools
1. Make a directory 
    <code>mkdir MSE544-Hyperdrive</code>
2. Move into the new directory
    <code>cd MSE544-Hyperdrive</code>
3. Clone the repository we will be using
    <code>git clone https://github.com/lunayuehuang/MSE544-HyperDrive.git</code>
4. Insert the following lines into "main.py" beginning on line 23
    <code>from azureml.core import Run
    run = Run.get_context()</code>
5. Insert the following line in "main.py" right before the "else" statement in line 196
    <code>run.log("MAE", np.float(mae_error.item()))</code>

## Part II: Setting up the Notebook
1. Make a jupyter notebook called "hyperdrive_experiment"
2. Insert a cell with the following imports
    <code>from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset, Run
    import azureml
    import os, tempfile, tarfile
    from azureml.train.hyperdrive import GridParameterSampling
    from azureml.train.hyperdrive import normal, uniform, choice
    from azureml.core.run import Run
    from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal</code>
3. Initialize a workspace in the next cell (be sure to enter the appropriate information)
    <code>subscription_id = <insert here>
    resource_group  = <insert here>
    workspace_name  = <insert here>
    ws = Workspace(subscription_id, resource_group, workspace_name)
    experiment = Experiment(workspace=ws, name='hyperdrive_experiment')</code>
4. Create a "dataset" variable to represent the .cif files we will use for training - THIS NEEDS TO BE UPDATED TO WORK WITH A KEY
    <code>dataset = Dataset.get_by_name(ws, name='materials_project_3207_unzipped')</code>
5. Set an environment variable using the repository .yml file
    <code>cgcnn_env = Environment.from_conda_specification(name='cgcnn_env', file_path='cgcnn_env.yml')</code>
6. Configurate the base training session
    <code>config = ScriptRunConfig(source_directory='./',   
                             script='main-hyper.py',       
                             compute_target='GPU-awoodwa', 
                             environment=cgcnn_env,
                             arguments=[
                                '--epochs', 5,
                                '--train-ratio', 0.6,
                                '--val-ratio', 0.2,
                                '--test-ratio', 0.2, 
                                 dataset.as_named_input('input').as_mount()]                   
                             )</code>
7. Define the parameters you are interested in sampling
    <code>param_sampling = GridParameterSampling( {
            "batch-size": choice(16, 64, 128),
            "n-conv": choice(1, 2, 3, 4, 5)
        }
    )</code>
8. Configure the hyperdrive experiment
    <code>hd_config = HyperDriveConfig(run_config=config,
                                 hyperparameter_sampling=param_sampling,
                                 primary_metric_name="MAE",
                                 primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                 max_total_runs=8,
                                 max_concurrent_runs=4)</code>
9. Finally, run the experiment and monitor the progress at the printed url
    <code>run = experiment.submit(hd_config)
    aml_url = run.get_portal_url()
    print(aml_url)</code>

## Part III: Running the Experiment
1. (Instructions for NAVIGATING AROUND AZURE)
# References