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

def get_accuracy(df, results, LR= False, atlas=None):
    if not LR:
        IDd = df.loc[df['ID']!='',:]
        corr1 = df.loc[df['ID']==results['autoID_1']]
        corr2 = df.loc[df['ID']==results['autoID_2']]
        corr3 = df.loc[df['ID']==results['autoID_3']]
        corr4 = df.loc[df['ID']==results['autoID_4']]
            
        corr_cum_2 = pd.concat([corr1,corr2]).drop_duplicates().reset_index(drop=True)
        corr_cum_3 = pd.concat([corr_cum_2,corr3]).drop_duplicates().reset_index(drop=True)
        corr_cum_4 = pd.concat([corr_cum_3,corr4]).drop_duplicates().reset_index(drop=True)

        per_ID = len(IDd.index)/len(df.index)
        per_corr_1 = len(corr1.index)/len(IDd.index)
        per_corr_2 = len(corr_cum_2.index)/len(IDd.index)
        per_corr_3 = len(corr_cum_3.index)/len(IDd.index)
        per_corr_4 = len(corr_cum_4.index)/len(IDd.index)

    else:
        atlas_df = atlas.get_df()

        df = pd.merge(df, atlas_df[['ID', 'neuron_class']], on='ID', how='left')

        merged_df = pd.merge(results, atlas_df[['ID', 'neuron_class']], left_on='autoID_1', right_on='ID', how='left')
        merged_df.rename(columns={'neuron_class': 'auto_class_1'}, inplace=True)
        merged_df.drop('ID', axis=1, inplace=True)

        merged_df = pd.merge(merged_df, atlas_df[['ID', 'neuron_class']], left_on='autoID_2', right_on='ID', how='left')
        merged_df.rename(columns={'neuron_class': 'auto_class_2'}, inplace=True)
        merged_df.drop('ID', axis=1, inplace=True)

        merged_df = pd.merge(merged_df, atlas_df[['ID', 'neuron_class']], left_on='autoID_3', right_on='ID', how='left')
        merged_df.rename(columns={'neuron_class': 'auto_class_3'}, inplace=True)
        merged_df.drop('ID', axis=1, inplace=True)

        # Merging for ID_4
        merged_df = pd.merge(merged_df, atlas_df[['ID', 'neuron_class']], left_on='autoID_4', right_on='ID', how='left')
        merged_df.rename(columns={'neuron_class': 'auto_class_4'}, inplace=True)
        merged_df.drop('ID', axis=1, inplace=True)

        IDd = df.loc[df['ID']!='',:]
        corr1 = df.loc[df['neuron_class']==merged_df['auto_class_1']]
        corr2 = df.loc[df['neuron_class']==merged_df['auto_class_2']]
        corr3 = df.loc[df['neuron_class']==merged_df['auto_class_3']]
        corr4 = df.loc[df['neuron_class']==merged_df['auto_class_4']]
            
        corr_cum_2 = pd.concat([corr1,corr2]).drop_duplicates().reset_index(drop=True)
        corr_cum_3 = pd.concat([corr_cum_2,corr3]).drop_duplicates().reset_index(drop=True)
        corr_cum_4 = pd.concat([corr_cum_3,corr4]).drop_duplicates().reset_index(drop=True)

        per_ID = len(IDd.index)/len(df.index)
        per_corr_1 = len(corr1.index)/len(IDd.index)
        per_corr_2 = len(corr_cum_2.index)/len(IDd.index)
        per_corr_3 = len(corr_cum_3.index)/len(IDd.index)
        per_corr_4 = len(corr_cum_4.index)/len(IDd.index)

    return IDd, per_ID, [per_corr_1, per_corr_2, per_corr_3, per_corr_4]