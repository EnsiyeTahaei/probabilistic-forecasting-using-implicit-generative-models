import datasets
from aae import AAE
from vae import VAE
from gan import GAN
from deepar import DeepAR
from utils import prepare_parser


def main():
    # Configuring arguments
    parser = prepare_parser()
    config = vars(parser)

    # Load dataset
    print("-- Loading dataset")
    dataset_x, dataset_y = datasets.load_dataset(dataset_name=config['dataset_name'],
                                                 condition_size=config['condition_size'],
                                                 horizon=config['horizon'],
                                                 val_rate=config['val_rate'],
                                                 test_rate=config['test_rate'],
                                                 dataset_dir=config['dataset_dir'])
    if config['dataset_name'] == "lorenz":
        x_test, y_test = datasets.load_dataset_per_condition(dataset_dir=config['dataset_dir'])

    # Declare operation mode
    if config['mode'] == 0:
        import tuneInterface
        print("**********  Mode: Tuning  *********")
        print("***********************************")
        tuneInterface.run(config)
    else:
        print("-- Loading model")
        if config['model_name'] == 'AAE':
            model = AAE(config, dataset_x, dataset_y)
        elif config['model_name'] == 'VAE':
            model = VAE(config, dataset_x, dataset_y)
        elif config['model_name'] == 'GAN':
            model = GAN(config, dataset_x, dataset_y)
        elif config['model_name'] == 'DeepAR':
            model = DeepAR(config, dataset_x, dataset_y)
        else:
            raise Exception("-- Warning: invalid model!!!!")

        if config['mode'] == 1:
            print("*********  Mode: Training  ********")
            print("***********************************")
            model.train()
            model.test()
            if config['dataset_name'] == "lorenz":
                model.test_on_lorenz(x_test=x_test, y_test=y_test)
        elif config['mode'] == 2:
            print("********  Mode: Forecasting  *******")
            print("***********************************")
            model.forecast()
        else:
            raise Exception("-- Warning: invalid mode!!!!")


if __name__ == '__main__':
    main()
