import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from utils import covar_to_coord
from stats import get_accuracy


def plot_summary_stats(segs, IDs, labels):

    data_seg = np.hstack(tuple(seg for seg in segs))
    data_ID = np.hstack(tuple(ID for ID in IDs))
    labels = np.hstack(tuple([labels[i]]*len(segs[i]) for i in range(len(labels))))
    #data_seg = np.hstack((num_seg_yem, num_seg_FOCO))
    #data_ID = np.hstack((num_ID_yem, num_ID_FOCO))
    #labels = np.hstack((np.asarray(['Yemini']*len(num_seg_yem)), np.asarray(['FOCO']*len(num_seg_FOCO))))

    df = pd.DataFrame({'seg':data_seg, 'ID':data_ID, 'labels':labels})

    print(df.head())

    fig, axs = plt.subplots(1,2, figsize=(12,6), sharey=True)
    sns.boxplot(ax = axs[0], data = df, x= 'labels', y='seg', orient='v')
    sns.boxplot(ax = axs[1], data = df, x= 'labels', y='ID', orient='v')
    axs[0].set_title('Number of segmented neurons in each dataset')
    axs[1].set_title('Number of IDd neurons in each dataset')
    axs[1].set_ylabel('')
    axs[0].set_ylabel('Number of neurons',fontsize=14)
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')

    plt.tight_layout()
    plt.show()

def plot_std_heatmap(num_heatmap, std_heatmap, df_ganglia):

    ganglia_indices = {}

    for ganglion in df_ganglia['ganglion'].unique():
        # Find the indices where the category starts and ends
        start_index = df_ganglia.index[df_ganglia['ganglion'] == ganglion][0]
        end_index = df_ganglia.index[df_ganglia['ganglion'] == ganglion][-1]
        
        # Store the start and end indices in the dictionary
        ganglia_indices[ganglion] = (start_index, end_index)

    # Create the figure and ax objects
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')

    mask = np.where(num_heatmap < 0.4, True, False)

    # Create the heatmap using the ax object
    sns.heatmap(std_heatmap, cmap='Reds')

    highlight_boxes = [((ganglia_indices[gang][0], ganglia_indices[gang][0]), (ganglia_indices[gang][1], ganglia_indices[gang][1]), gang) for gang in df_ganglia['ganglion'].unique()]

    # Overlay boxes
    for (x1, y1), (x2, y2), label in highlight_boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='black', lw=3))
        if x1<len(df_ganglia)/2:
            plt.text(x2+2, (y1 + y2) / 2, label, color='black', ha='left', va='center')
        else:
            plt.text(x1-2, (y1 + y2) / 2, label, color='black', ha='right', va='center')

    ax.set_title('Standard deviation of pairwise distances')
    plt.tick_params(which='both', bottom=False, left=False,labelbottom=False, labelleft=False)  # Hide tick labels
    plt.show()

def plot_num_heatmap(num_heatmap, df_ganglia):
    ganglia_indices = {}

    for ganglion in df_ganglia['ganglion'].unique():
        # Find the indices where the category starts and ends
        start_index = df_ganglia.index[df_ganglia['ganglion'] == ganglion][0]
        end_index = df_ganglia.index[df_ganglia['ganglion'] == ganglion][-1]
        
        # Store the start and end indices in the dictionary
        ganglia_indices[ganglion] = (start_index, end_index)

    # Create the figure and ax objects
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')

    mask = np.where(num_heatmap < 0.4, True, False)

    # Create the heatmap using the ax object
    sns.heatmap(num_heatmap, cmap='Reds')

    highlight_boxes = [((ganglia_indices[gang][0], ganglia_indices[gang][0]), (ganglia_indices[gang][1], ganglia_indices[gang][1]), gang) for gang in df_ganglia['ganglion'].unique()]

    # Overlay boxes
    for (x1, y1), (x2, y2), label in highlight_boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='black', lw=3))
        if x1<len(df_ganglia)/2:
            plt.text(x2+2, (y1 + y2) / 2, label, color='black', ha='left', va='center')
        else:
            plt.text(x1-2, (y1 + y2) / 2, label, color='black', ha='right', va='center')

    ax.set_title('Fraction of datasets each pairing appears in')
    plt.tick_params(which='both', bottom=False, left=False,labelbottom=False, labelleft=False)  # Hide tick labels
    plt.show()

def plot_atlas_unrolled(df):
    """df needs: x/y/zcyl, ganglion, h, theta """

    ganglia = sorted(df['ganglion'].unique())
    gs_kw = dict(height_ratios=[1, 4])
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6, 12), gridspec_kw=gs_kw)

    for g in ganglia:
        dfg = df[df['ganglion'] == g]
        ax[0].plot(dfg['ycyl'], dfg['zcyl'], 'o', lw=0,markerfacecolor='None')
        ax[1].plot(dfg['theta'], dfg['h'], 'o', lw=0, label=g, markerfacecolor='None')

    ax[0].set_aspect('equal')
    ax[0].set_xlim([-4, 4])
    ax[0].set_ylim([-4, 4])
    ax[0].plot([0, 0], [0, 2.5], '--', color='grey')
    ax[0].plot(0, 0, 'x', color='k')

    ax[1].axvspan(-135, -45, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvspan(45, 135, edgecolor=None, color='lightgrey', alpha=0.4, zorder=0, lw=0)
    ax[1].axvline(-180, ls='--', color='grey')
    ax[1].axvline(180, ls='--', color='grey')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def plot_atlas_RGB(df, sigma):

    rgb_mu = np.asarray(df[['R', 'G', 'B']])
    rgb_sigma = sigma[3:7,3:7, :]

    fig, axs = plt.subplots(1 , 3, figsize=(24,48))

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-10, 30)
        ax.set_ylim(-10, 30)

    for n in range(rgb_sigma.shape[2]):
        
        rgl1, rgl2, rgtheta = covar_to_coord(rgb_sigma[[0,1],:,n][:,[0,1]])
        rbl1, rbl2, rbtheta = covar_to_coord(rgb_sigma[[0,2],:,n][:,[0,2]])
        gbl1, gbl2, gbtheta = covar_to_coord(rgb_sigma[[1,2],:,n][:,[1,2]])

        rmu = rgb_mu[n, 0]
        gmu = rgb_mu[n, 1]
        bmu = rgb_mu[n, 2]
        
        #looking at only half a std to make it easier to visualize 

        rg_ellipse = Ellipse((rmu,gmu), width =rgl1*2, height = rgl2*2, angle=rgtheta*180/np.pi, alpha=0.05, edgecolor='orange', facecolor='orange')
        axs[0].add_patch(rg_ellipse)
        rb_ellipse = Ellipse((rmu, bmu), width =rbl1*2, height = rbl2*2, angle=rbtheta*180/np.pi, alpha=0.05, edgecolor='magenta', facecolor='magenta')
        axs[1].add_patch(rb_ellipse)
        gb_ellipse = Ellipse((gmu, bmu), width =gbl1*2, height = gbl2*2, angle=gbtheta*180/np.pi, alpha=0.05, edgecolor='cyan', facecolor='cyan')
        axs[2].add_patch(gb_ellipse)

    axs[0].set_title('red-green')
    axs[0].set_xlabel('red')
    axs[0].set_ylabel('green')
    axs[1].set_title('red-blue')
    axs[1].set_xlabel('red')
    axs[1].set_ylabel('blue')
    axs[2].set_title('green-blue')
    axs[2].set_xlabel('green')
    axs[2].set_ylabel('blue')


    plt.show()

def plot_atlas_2d_views(df_atlas, sigma, df_data):

    xyz_data = df_data[['X', 'Y', 'Z']]
    rgb_data = np.asarray(df_data[['R', 'G', 'B']])

    xyz_sigma = sigma[0:3, 0:3,:]

    fig = plt.figure(figsize=(15,6))
    ax = [plt.subplot(211)]
    ax.append(plt.subplot(212, sharex=ax[0]))

    #ax[0].plot(df_atlas['X'], df_atlas['Z'], 'o', mec='grey', ms=15)
    for i, row in df_atlas.iterrows():
        xzl1, xzl2, xztheta = covar_to_coord(xyz_sigma[[0,2],:,i][:,[0,2]]) 
        xyl1, xyl2, xytheta = covar_to_coord(xyz_sigma[[0,1],:,i][:,[0,1]])

        xz_ellipse = Ellipse((row['X'], row['Z']), width = xzl1*2, height=xzl2*2, angle=xztheta*180/np.pi, alpha=0.2, edgecolor = 'blue', facecolor='blue', linestyle='-')
        xy_ellipse = Ellipse((row['X'], row['Y']), width = xyl1*2, height=xyl2*2, angle=xytheta*180/np.pi, alpha=0.2, edgecolor = 'blue', facecolor='blue', linestyle='-')
        ax[0].add_patch(xz_ellipse)
        ax[1].add_patch(xy_ellipse)
        
    colors_min = np.amin(rgb_data, axis=0)
    colors_max = np.amax(rgb_data, axis=0)
    color_norm = np.divide(rgb_data-colors_min, colors_max-colors_min)
  
        
    ax[0].scatter(xyz_data['X'], xyz_data['Z'], c=color_norm)
    ax[1].scatter(xyz_data['X'], xyz_data['Y'], c=color_norm)
  
    ax[0].set_aspect('equal')
    ax[0].grid()
    ax[0].set_ylabel('Z')
    ax[0].autoscale_view()

    ax[1].set_aspect('equal')
    ax[1].grid()
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].autoscale_view()

    #plt.tight_layout()
    plt.show()


def plot_color_discrim(datasets, labels):

    data = {}

    for i in range(len(datasets)):
        category = labels[i]
        data[category] = np.concatenate([val for val in datasets[i].values()])

    series_list = [pd.Series(data[key],name=key) for key in data]

    df = pd.concat(series_list, axis=1)

    melted = pd.melt(df, var_name = 'dataset', value_name='avg_col_dist')

    sns.set(style='whitegrid')

    sns.catplot(x='dataset', y='avg_col_dist', data=melted, kind='violin')

    plt.xlabel('Dataset')
    plt.ylabel('Average normalized color distance to neighbors')
    plt.title('Weighted average of color discriminability to 6 nearest neighbors of each neuron')

    plt.show()

def plot_neur_nums(neurons, num_datasets, atlas):

    neur_df = atlas.df[['ID', 'ganglion']]

    dict_df = pd.DataFrame(list(neurons.items()), columns = ['ID', 'num'])
    dict_df['frac'] = dict_df['num']/num_datasets
 
    merged = pd.merge(neur_df, dict_df, on='ID') #this will preserve the order of neurons from neur_df which is sorted by ganglion and then distance along x axis

    sns.set(style='whitegrid')

    plt.figure(figsize=(12,8))
    sns.barplot(x='ID', y='frac', hue='ganglion', data=merged)

    plt.xlabel('Neuron IDs')
    plt.ylabel('Fraction of datasets with ground truth labeled neuron')

    plt.xticks(rotation=60, ha='right', fontsize=6)

    bar_width = 0.7

    for patch in plt.gca().patches:
        current_width = patch.get_width()
        diff = current_width - bar_width

        # Change the bar width
        patch.set_width(bar_width)

        # Recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

    # Show the plot
    plt.legend(title='ganglion', loc='upper right')

    plt.show()

def plot_accuracies(datasets, csv_folders, labels, LR=False, atlas=None, title = 'Accuracy by dataset'):

    num_data_total = np.sum([len(dataset.keys()) for dataset in datasets])

    accs = np.empty((num_data_total, 6), dtype='O')

    k=0

    for i, dataset in enumerate(datasets):
        csv_folder = csv_folders[i]
        label = labels[i]
        for j, key in enumerate(dataset.keys()):
            csv = csv_folder + key +'.csv'
            IDd, per_ID, acc = get_accuracy(dataset[key], pd.read_csv(csv), LR=LR, atlas=atlas)
            acc.append(label)
            acc.append(key)
            accs[k,:] = acc
            k+=1

    df = pd.DataFrame(accs, columns = ['top_acc', 'top_2_acc', 'top_3_acc', 'top_4_acc', 'dataset', 'identifier'])

    g = sns.catplot(data=df, kind='box', x = 'dataset', y='top_acc', hue='dataset', orient='v', dodge=False)

    plt.ylabel('Percent autoID accuracy')
    plt.ylim((0,1))

    plt.title('NeuroPAL AutoID accuracy by dataset')

    plt.show()

    return df