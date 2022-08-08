"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import processingData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def geneticMain(inpath):
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of networks in each generation.

    data = processingData.processingData()
    raw_dataset = data.csvProcess(filepath=inpath, config=3)
    binEdges = data.binEdges
    dataset = raw_dataset.drop(
        columns=['ematches', 'genElectronCharge', 'electronCharge', 'genMuonCharge', 'muonCharge', 'electron_eta'])
    test_dataset = dataset.sample(frac=0.3, random_state=0)
    train_dataset = dataset.drop(test_dataset.index)

    num_events_per_training_bin = 10000  # for equally weighting all pt bins in training
    genMuonDist = plt.hist(np.array(train_dataset['mpT']), binEdges)
    histScale = num_events_per_training_bin * 0.8 / genMuonDist[0]
    histScaleFloor = np.floor(histScale)
    data_list = []
    for index, row in train_dataset.iterrows():
        for i in range(len(binEdges) - 1):
            if binEdges[i] < row['mpT'] < binEdges[i + 1]:
                if histScale[i] > 1:
                    for j in range(int(histScaleFloor[i])):
                        data_list.append(row)
                    if np.random.rand() < histScale[i] - histScaleFloor[i]:
                        data_list.append(row)
                else:
                    if np.random.rand() < histScale[i]:
                        data_list.append(row)

    train_dataset = pd.DataFrame(data_list, columns=train_dataset.columns)

    train_dataset = train_dataset.sample(frac=1)
    train_dataset.to_csv('raw_train_dataset.csv', index=False)
    test_dataset = test_dataset.sample(frac=1)
    test_dataset.to_csv('raw_test_dataset.csv', index=False)

    nn_param_choices = {
        'nb_neurons': [32, 64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN options')
    parser.add_argument('--in_path', type=str, default='neuralNetDataTT_sum.csv', help='path to the csv input file')
    args = parser.parse_args()
    geneticMain(args.in_path)
