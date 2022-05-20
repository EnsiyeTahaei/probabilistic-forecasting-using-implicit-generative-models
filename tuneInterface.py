import gc
import os

import joblib
import optuna
import torch
import torch.multiprocessing as mp
from optuna.trial import TrialState

import utils


def run(config):
    """
    Tuning interface module
    :param config: config information
    """
    mp.set_start_method('spawn')
    torch.cuda.empty_cache()
    gc.collect()

    if config['model_name'] == 'AAE':
        if config['tune_cell'] == 'RNN':
            from tuning.tuning_aae_rnn import tune
        elif config['tune_cell'] == 'TCN':
            from tuning.tuning_aae_tcn import tune
    elif config['model_name'] == 'VAE':
        if config['tune_cell'] == 'RNN':
            from tuning.tuning_vae_rnn import tune
        elif config['tune_cell'] == 'TCN':
            from tuning.tuning_vae_tcn import tune
    elif config['model_name'] == 'GAN':
        if config['tune_cell'] == 'RNN':
            from tuning.tuning_gan_rnn import tune
        elif config['tune_cell'] == 'TCN':
            from tuning.tuning_gan_tcn import tune
    elif config['model_name'] == 'DeepAR':
        if config['tune_cell'] == 'RNN':
            from tuning.tuning_deepar_rnn import tune
        elif config['tune_cell'] == 'TCN':
            from tuning.tuning_deepar_tcn import tune
    else:
        raise Exception("-- Warning: invalid model!!!!")

    tune_dir = os.path.join(config['result_dir'], "tuning")
    tune_dir = os.path.join(tune_dir, config['tune_cell'])
    utils.check_dir(tune_dir)

    params = {'max_steps': config['max_steps'],
              'batch_size': config['batch_size'],
              'dataset_name': config['dataset_name'],
              'dataset_dir': config['dataset_dir'],
              'val_rate': config['val_rate'],
              'test_rate': config['test_rate'],
              'seed': config['seed'],
              'tune_dir': tune_dir}

    database = "tuning_" + config['model_name'] + "_" + config['dataset_name'] + "_" + config['tune_cell']
    storage = "sqlite:///tuningDB.db"
    optuna.create_study(study_name=database,
                        direction="minimize",
                        load_if_exists=True,
                        storage=storage)

    devices = []
    for i in range(config['max_device']):
        devices.append("cuda:" + str(i))
    process_per_device = config['process_per_device']
    available_devices = devices * process_per_device
    # available_devices = ["cpu"]

    processes = []
    for device in available_devices:
        p = mp.Process(target=tune, args=(device, database, storage, params,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    study = optuna.load_study(study_name=database, storage=storage)
    joblib.dump(study, tune_dir + '/' + database + '.pkl')

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("*********  Study statistics  ********")
    print("*************************************")
    print("-- Number of all trials: ", len(study.trials))
    print("-- Number of pruned trials: ", len(pruned_trials))
    print("-- Number of complete trials: ", len(complete_trials))

    print("-- Best trial:")
    trial = study.best_trial
    print("-- Value: ", trial.value)
    print("-- Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
