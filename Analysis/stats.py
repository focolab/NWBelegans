import numpy as np
import pandas as pd

def get_summary_stats(dataset):
    num_segmented = []
    num_ID = []

    for file, blobs in dataset.items():
        IDd = blobs[blobs['ID']!='']

        num_segmented.append(len(blobs))
        num_ID.append(len(IDd))

    return num_segmented, num_ID

def analyze_pairs(pairings, neuron_ganglia, num_datasets):
    num_pairings = len(pairings.keys())/2
    neurons = np.asarray(neuron_ganglia['ID'])

    num_heatmap = np.zeros((len(neurons),len(neurons)))
    std_heatmap = np.zeros((len(neurons), len(neurons)))

    for i in range(len(neurons)):
        for j in range(i, len(neurons)):

            label1 = neurons[i]
            label2 = neurons[j]
            pair = label1 + '-' +label2
            pair2 = label2 + '-' + label1

            if i == j and pair in pairings:
                num_heatmap[i,j] = len(pairings[pair])
                std_heatmap[i,j] = np.std(pairings[pair])
            if pair in pairings:
                num_heatmap[i,j] = len(pairings[pair])
                num_heatmap[j,i] = len(pairings[pair])
                std_heatmap[i,j] = np.std(pairings[pair])
                std_heatmap[j,i] = np.std(pairings[pair])

    return num_pairings, num_heatmap/num_datasets, std_heatmap