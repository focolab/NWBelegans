# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:31:11 2020

@author: Amin
"""
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
from skimage import color
from . import Helpers
import pandas as pd
import scipy as sp
import numpy as np
import copy


# %%
class Atlas:
    """Class for training an atlas from multiple annotated images by aligning 
        the images using affine transformations
    """
    
    iter        = 10
    min_counts  = 2
    epsilon     = [1e+3,1e+3]
    
    @staticmethod
    def axis_equal(ax):
        """Equalize axes of a 3D plot
    
        Args:
            ax (matplotlib.axis): Axis to be equalized 
            
        """
        
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    
    @staticmethod
    def visualize(atlas,aligned,title_str,fontsize=9,dotsize=30,save=False,file=None):
        """Visualize trained atlas and aligned point clouds
    
        Args:
            atlas
            aligned
            title_str
            fontsize
            dotsize
            save
            file
            
        """
        
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_title(title_str)

        atlas_color = atlas['mu'][:,3:]
        atlas_color[atlas_color<0] = 0
        atlas_color = atlas_color/atlas_color.max()
        if atlas_color.shape[1] == 4:
            atlas_color[:,3] = 0
        
        sizes = 1*np.array([np.linalg.eig(atlas['sigma'][:3,:3,i])[0].sum() for i in range(atlas['sigma'].shape[2])])
        
        ax.scatter(atlas['mu'][:,0],atlas['mu'][:,1],atlas['mu'][:,2], s=dotsize, facecolors=atlas_color[:,:3], marker='.')
        ax.scatter(atlas['mu'][:,0],atlas['mu'][:,1],atlas['mu'][:,2], s=sizes, facecolors=atlas_color, edgecolors=atlas_color[:,:3], marker='o',linewidth=1)
        
        
        for i in range(len(atlas['names'])):
            ax.text(atlas['mu'][i,0],atlas['mu'][i,1],atlas['mu'][i,2],atlas['names'][i],c=atlas_color[i,:3],fontsize=fontsize)

        for j in range(len(aligned)):
            al = aligned[j]
            c_j = al[:,3:6]
            c_j[c_j < 0] = 0
            ax.scatter(al[:,0],al[:,1],al[:,2], s=10, c=c_j/c_j.max(),marker='.');
        
        ax.set_ylim([35+atlas['mu'][:,1].min(),-35+atlas['mu'][:,1].max()])
        Atlas.axis_equal(ax)
        ax.view_init(elev=90., azim=10)
        
        
        ax.axis('off')
        ax.set_facecolor('xkcd:light gray')
        fig.patch.set_facecolor('xkcd:light gray')
        
        if save:
            plt.savefig(file+'.png',format='png')
            try:
                plt.savefig(file+'.pdf',format='pdf')
            except:
                pass
            plt.close('all')
        else:
            plt.show()
            
            
    def major_axis_align(atlas,aligned,params,shift=0):
        pca = PCA(n_components=3)
        pca.fit(atlas['mu'][:,:3])
        projection = pca.components_.T
        projection = projection*np.linalg.det(projection)
        projection = projection[:,[1,0,2]]
        
        shift = -(atlas['mu'][:,:3]@projection).min(0)+shift
        
        mus = atlas['mu'][:,:3]@projection + shift
        cov = np.zeros((3,3,mus.shape[0]))
        for i in range(cov.shape[2]):
            cov[:,:,i] = projection.T@atlas['sigma'][:3,:3,i]@projection
            
        
        proj_atlas = copy.deepcopy(atlas)
        proj_atlas['mu'][:,:3] = mus
        proj_atlas['sigma'][:3,:3,:] = cov
        
        proj_params = copy.deepcopy(params)
        for i in range(params['beta'].shape[2]):
            proj_params['beta'][:3,:3,i] = proj_params['beta'][:3,:3,i]@projection
            proj_params['beta0'][:,:3,i] = proj_params['beta0'][:,:3,i]@projection+shift
        
        proj_aligned = copy.deepcopy(aligned)
        for i in range(len(aligned)):
            proj_aligned[i][:,:3] = aligned[i][:,:3]@projection+shift
        
        return proj_atlas,proj_aligned,proj_params

    def visualize_pretty(atlas,aligned,title_str,projection=None,fontsize=8,dotsize=12,save=False,file=None,olp=False,tol=1e0,plot_cov=True,hsv_correction=False,connect_pairs=None,alpha=1,labels=None):
        """Visualize trained atlas and aligned point clouds
    
        Args:
            atlas
            aligned
            title_str
            fontsize
            dotsize
            save
            file
            olp: optimal label positioning
            
        """
        
        fig = plt.figure(figsize=(18,4))

        ax = fig.add_subplot(111)
        
        ax.set_title(title_str)

        atlas_color = atlas['mu'][:,3:].copy()
        
        atlas_color[atlas_color<0] = 0
        atlas_color = atlas_color/(np.percentile(atlas_color,95,axis=0)+1e-5)
        atlas_color[atlas_color>1] = 1
        
        if hsv_correction:
            hsv = color.rgb2hsv(atlas_color[None,:,:3]).squeeze()
            hsv[:,2] = 1
            atlas_color[:,:3] = color.hsv2rgb(hsv[None,:,:]).squeeze()
        
        if atlas_color.shape[1] == 4:
            atlas_color[:,3] = alpha
        
        if projection is None:
            pca = PCA(n_components=2)
            pca.fit(atlas['mu'][:,:3])
            projection = pca.components_.T
        
        mus = atlas['mu'][:,:3].copy()@projection
        cov = np.zeros((2,2,mus.shape[0]))
        for i in range(cov.shape[2]):
            cov[:,:,i] = projection.T@atlas['sigma'][:3,:3,i].copy()@projection
            
        samples = [[]]*len(aligned)
        for i in range(len(aligned)):
            samples[i] = aligned[i][:,:3]@projection
            
        if plot_cov:
            for i in range(cov.shape[2]):
                Atlas.draw_ellipse(mus[i,:],cov[:,:,i],atlas_color[i,:3][None,:],
                                   std_devs=1.5,ax=ax,line_width=2)

        ax.scatter(mus[:,0],mus[:,1],facecolors=atlas_color, s=300,
                   edgecolors='k',marker='.',linewidth=1)
        
        if olp:
            if labels is None:
                label_coor = Atlas.optimal_label_positioning(mus[:,:2],tol=tol)
            else:
                label_coor = mus[:,:2].copy()
                label_coor[labels,:] = Atlas.optimal_label_positioning(mus[labels,:2],tol=tol)
        else:
            label_coor = mus[:,:2].copy()
        
        for i in range(len(atlas['names'])):
            if labels is None or labels is not None and labels[i]:
                ax.text(label_coor[i,0],label_coor[i,1],atlas['names'][i],
                        c=atlas_color[i,:3],fontsize=fontsize)
            ax.plot([label_coor[i,0], mus[i,0]],[label_coor[i,1], mus[i,1]],color=atlas_color[i,:3],linestyle='dotted',linewidth=1)
        ax.set_xlim(label_coor.min(0)[0]-5,label_coor.max(0)[0]+5)
        ax.set_ylim(label_coor.min(0)[1]-5,label_coor.max(0)[1]+5)
        
        if connect_pairs is not None:
            for pair in connect_pairs:
                ax.plot([mus[pair[0],0], mus[pair[1],0]],
                        [mus[pair[0],1], mus[pair[1],1]],
                        color='k',linestyle='dotted',linewidth=1)
        
        for j in range(len(samples)):
            c_j = aligned[j][:,3:6]
            c_j[c_j < 0] = 0
            c_j = c_j/c_j.max()
            ax.scatter(samples[j][:,0],samples[j][:,1], s=dotsize, 
                       facecolors=c_j, edgecolors=c_j, marker='.')
        
        ax.set_aspect('equal',adjustable='box')

        # ax.grid('on')
        ax.set_facecolor('xkcd:light gray')
        fig.patch.set_facecolor('xkcd:light gray')
        fig.tight_layout()

        if save:
            plt.savefig(file+'.png',format='png')
            try:
                plt.savefig(file+'.pdf',format='pdf')
            except:
                pass
            plt.close('all')
        else:
            plt.show()

        
    @staticmethod
    def draw_ellipse(mean,covariance,color,std_devs=3,ax=None,line_width=2):
        # sample grid that covers the range of points
        min_p = mean - std_devs*np.sqrt(np.diag(covariance))
        max_p = mean + std_devs*np.sqrt(np.diag(covariance))
        
        x = np.linspace(min_p[0],max_p[0],256) 
        y = np.linspace(min_p[1],max_p[1],256)
        X,Y = np.meshgrid(x,y)
        
        Z = multivariate_normal.pdf(np.stack((X.reshape(-1),Y.reshape(-1))).T, mean=mean, cov=(std_devs**2)*covariance)
        Z = Z.reshape([len(x),len(y)])
        
        if ax is None:
            plt.contour(X, Y, Z, 0,colors=color,linewidth=line_width)
        else:
            ax.contour(X, Y, Z, 0,colors=color,linewidths=line_width)


    @staticmethod
    def update_beta(X,model):
        """Updating the transformation parameters for different images
    
        Args:
            X (numpy.ndarray): 
            model (dict):
            
        Returns:
            params (dict):
            aligned (numpy.ndarray):
        """
        # allocating memory
        beta    = np.zeros((X.shape[1],X.shape[1],X.shape[2]))
        beta0   = np.zeros((1,X.shape[1],X.shape[2]))
        aligned = np.zeros(X.shape)
        
        params = {}
        
        C = X.shape[1]-3
        
        # computing beta for each training worm based using multiple
        # covariance regression solver
        
        cost = [0,0]

        sigma_inv = np.array([np.linalg.inv(model['sigma'][:,:,i]) 
                          for i in range(model['sigma'].shape[2])]).transpose([1,2,0])
        
        for j in range(X.shape[2]):
            idx = ~np.isnan(X[:,:,j]).all(1)
            
            # solving for positions
            R = Helpers.MCR_solver(model['mu'][idx,:3], \
                np.concatenate((X[idx,:3,j], np.ones((idx.sum(), 1)) ),1), \
                model['sigma'][:3,:3,idx])

            beta[:3,:3,j] = R[:3,:3]
            beta0[:,:3,j] = R[3,None]
            
            # solving for colors
            R = Helpers.MCR_solver(model['mu'][idx,3:], \
                np.concatenate((X[idx,3:,j], np.ones((idx.sum(),1)) ), 1), \
                model['sigma'][3:,3:,idx])

            beta[3:,3:,j] = R[:C,:C]
            beta0[:,3:,j] = R[C,None]
            
            aligned[:,:,j] = (X[:,:,j])@beta[:,:,j]+beta0[:,:,j]
            
            cost[0] += sum([sp.spatial.distance.mahalanobis(aligned[i,:3,j].squeeze(),model['mu'][i,:3].squeeze(),sigma_inv[:3,:3,i]) for i in np.where(idx)[0]])
            cost[1] += sum([sp.spatial.distance.mahalanobis(aligned[i,3:,j].squeeze(),model['mu'][i,3:].squeeze(),sigma_inv[3:,3:,i]) for i in np.where(idx)[0]])
        
        params['beta'] = beta
        params['beta0'] = beta0
        
        return params, aligned, cost
    
    @staticmethod
    def initialize_atlas(col,pos, match_indexes, train_indices):
        """Initialize atlas by finding the best image for aligning all other 
            images to
    
        Args:
            col (numpy.ndarray): 
            pos (numpy.ndarray):
                
        Returns:
            model (dict):
            params (dict):
            X (numpy.ndarray):
            
        """
		
        # memory allocation
        model = {}
        cost    = np.zeros((pos.shape[2],len(match_indexes)))
        aligned = [np.zeros((pos.shape[0], pos.shape[1]+col.shape[1],pos.shape[2]))]*len(match_indexes)
        
        # alignment of samples to best fit worm
        for i in range(pos.shape[2]):
            for j, num in enumerate(match_indexes):
                S0,R0,T0 = Helpers.scaled_rotation(pos[:,:,i],pos[:,:,num])
                cost[i,j] = np.sqrt(np.nanmean(np.nansum((pos[:,:,i]@(R0*S0)+T0-pos[:,:,num])**2,1),0))

                aligned[j][:,:3,i] = pos[:,:,i]@(R0*S0)+T0
                aligned[j][:,3:,i] = col[:,:,i]
        
        jidx = np.argmin(cost.sum(0))
        X = aligned[jidx]

        model['mu'] = np.nanmean(X[:,:,train_indices],2)   
        
        return model,X
    
    @staticmethod
    def estimate_mu(mu,aligned):
        """Estimate the neuron centers and colors using updated and aligned 
            images
    
        Args:
            mu (numpy.ndarray): Previous value of mu with size Nx(3+C)
            aligned (numpy.ndarray): Aligned point clouds with size Nx(3+C)xK
                
        Returns:
            mu (dict): Updated value of mu
            
        """
        
        # eigen-push to preserve the volume, computing singular values
        # before the update
        _,Sp,_ = np.linalg.svd(mu[:,:3]-mu[:,:3].mean(0))
        _,Sc,_ = np.linalg.svd(mu[:,3:]-mu[:,3:].mean(0))
        
        # computing the means
        mu = np.nanmean(aligned,2)
        
        # updaing using singular vectors of new means and singular
        # values of the old means
        Up,_,Vp = np.linalg.svd(mu[:,:3]-mu[:,:3].mean(0),full_matrices=False)
        mu[:,:3] = Up@np.diag(Sp)@Vp + mu[:,:3].mean(0)
        
        Uc,_,Vc = np.linalg.svd(mu[:,3:]-mu[:,3:].mean(0),full_matrices=False)
        mu[:,3:] = Uc@np.diag(Sc)@Vc + mu[:,3:].mean(0)
        
        return mu
        
    @staticmethod
    def estimate_sigma(mu,aligned,reg=[0,0]):
        """Estimate the covariance of position and color
    
        Args:
            mu (numpy.ndarray): Previous value of mu with size Nx(3+C)
            aligned (numpy.ndarray): Aligned point clouds with size Nx(3+C)xK
                
        Returns:
            sigma (dict): Updated value of covariance Nx(3+C)x(3+C)
            
        """
        
        # memory allocation
        sigma   = np.zeros((mu.shape[1],mu.shape[1],mu.shape[0]))
        
        # computing the covariances
        for i in range(aligned.shape[0]):
            sigma[:,:,i] = pd.DataFrame((aligned[i,:,:]).T).cov().to_numpy()
        
        # well-condition the sigmas by adding epsilon*I
        sigma[:3,:3,:] = sigma[:3,:3,:] + reg[0]*np.eye(sigma[:3,:3,:].shape[0])[:,:,None]
        sigma[3:,3:,:] = sigma[3:,3:,:] + reg[1]*np.eye(sigma[3:,3:,:].shape[0])[:,:,None]
        
        for i in range(aligned.shape[0]):
            # sigma[:,:,i] = np.eye(len(sigma))
            # decorrelate color and position 
            sigma[:3,3:,i] = 0
            sigma[3:,:3,i] = 0
            
            # diagonalize color covariances
            sigma[3:,3:,i] = np.diag(np.diag(sigma[3:,3:,i]))
        
        return sigma
    
    
    @staticmethod
    def sort_mu(ims, train_indices, neurons=None):
        
        annotations = [x.get_annotations() for x in ims]
        scales = [np.array([1,1,1]) for x in ims]
        positions = [x.get_positions(x.scale) for x in ims]
        colors = [x.get_colors_readout() for x in ims]
        
        # reading the annotations
        N = list(set([item for sublist in annotations for item in sublist]))
        
        C = colors[0].shape[1]
        # allocationg memory for color and position
        pos = np.zeros((len(N),3,len(annotations)))*np.nan
        col = np.zeros((len(N),C,len(annotations)))*np.nan
        
        # re-ordering colors and positions to have the same neurons in
        # similar rows
        for j in range(len(annotations)):
            perm = np.array([N.index(x) for x in annotations[j]])
            pos[perm,:,j] = positions[j]*scales[j][None,:]
            col_tmp = colors[j]
            col[perm,:,j] = col_tmp
        
        # computing the number of worms with missing data for each
        # neuron
        counts = (~np.isnan(pos[:,:,train_indices].sum(1))).sum(1)
        # filtering the neurons based on min_count of the missing data
        good_indices = np.logical_and( counts>Atlas.min_counts, 
                                      ~np.array([x == '' or x == None for x in N]))
        pos = pos[good_indices ,:,:]
        col = col[good_indices ,:,:]
        
        N = [N[i] for i in range(len(good_indices)) if good_indices[i]]
        
        if neurons is not None:
            idx = [i for i in range(len(N)) if N[i] in neurons]
            N = [N[i] for i in idx]
            pos = pos[idx,:]
            col = col[idx,:]
        
        return N,col,pos,counts
        
    
    @staticmethod
    def train_atlas(ims_,bodypart,neurons=None, match_indexes=[], train_indices=[]):
        """Main function for estimating the atlas of positions and colors
    
        Args:
            annotations (array): 
            scales (numpy.ndarray): 
            positions (numpy.ndarray):
            colors (numpy.ndarray):
            bodypart (string):
                
        Returns:
            atlas (dict):
            aligned_coord (numpy.ndarray):
            
        """

        train_indices = np.asarray(train_indices)
        
        ims = copy.deepcopy(ims_)
        bodypart = ims[0].bodypart
        
        for i in range(len(ims)):
            im = ims[i]
            if neurons is not None:
                im.neurons = [im.neurons[i] for i in range(len(im.neurons)) if im.neurons[i].annotation in neurons]
                
        annotations = [x.get_annotations() for x in ims]
        colors = [x.get_colors_readout() for x in ims]
                              
        C = colors[0].shape[1]

        N,col,pos,counts = Atlas.sort_mu(ims, train_indices)

        if match_indexes == []:
            match_indexes = range(len(ims))
        
        # initialization
        model,aligned = Atlas.initialize_atlas(col,pos, match_indexes, train_indices)

        init_aligned = np.hstack((pos,col))
        
        cost = []
        for iteration in range(Atlas.iter):
            # updating means
            model['mu'] = Atlas.estimate_mu(model['mu'],aligned[:,:,train_indices])
            
            # updating sigma
            model['sigma'] = Atlas.estimate_sigma(model['mu'],aligned[:,:,train_indices],reg=Atlas.epsilon)
            
            # updating aligned
            params,aligned,cost_ = Atlas.update_beta(init_aligned,model)
            cost.append(cost_)
            print('Iteration: ' + str(iteration) + ' - Cost (pos,col): ' + str(cost_))
            
        model['mu'] = Atlas.estimate_mu(model['mu'],aligned[:,:,train_indices])
        model['sigma'] = Atlas.estimate_sigma(model['mu'],aligned[:,:,train_indices])
        
        # store the result for the output
        atlas = {'bodypart':bodypart, \
                      'mu': model['mu'], \
                      'sigma': model['sigma'], \
                      'names': N,
                      'aligned': aligned}
        # store worm specific parameters inside their corresponding
        # class

        aligned_coord = []
        for j in range(len(annotations)):
            perm = np.array([N.index(x) if x in N else -1 for x in annotations[j]])
            aligned_coord.append(np.array([aligned[perm[n],:,j] if perm[n] != -1 else np.zeros((C+3))*np.nan for n in range(len(perm))]))
            
        return atlas,aligned_coord,params,cost,counts
    
    @staticmethod
    def train_distance_atlas(annotations,scales,positions,colors,bodypart):
        # reading the annotations
        N = list(set([item for sublist in annotations for item in sublist]))
        N.sort()
        
        C = colors[0].shape[1]
        # allocationg memory for color and position
        pos = np.zeros((len(N),3,len(annotations)))*np.nan
        col = np.zeros((len(N),C,len(annotations)))*np.nan
        
        # re-ordering colors and positions to have the same neurons in
        # similar rows
        for j in range(len(annotations)):
            perm = np.array([N.index(x) for x in annotations[j]])
            pos[perm,:,j] = positions[j]*scales[j][np.newaxis,:]
            col_tmp = colors[j]
            col[perm,:,j] = col_tmp
        
        counts = (~np.isnan(pos.sum(1))).sum(1)
        # filtering the neurons based on min_count of the missing data
        good_indices = np.logical_and( counts>Atlas.min_counts, 
                                      ~np.array([x == '' or x == None for x in N]))
        pos = pos[good_indices ,:,:]
        col = col[good_indices ,:,:]
        
        N = [N[i] for i in range(len(good_indices)) if good_indices[i]]
        
        D1 = np.zeros((len(N),len(N)))
        D2 = np.zeros((len(N),len(N)))
        
        C1 = np.zeros((len(N),len(N)))
        C2 = np.zeros((len(N),len(N)))
        
        for i in range(len(N)):
            for j in range(len(N)):
                D1[i,j] = np.nanmean(np.sqrt(((pos[i,:,:] - pos[j,:,:])**2).sum(0)))
                C1[i,j] = np.nanstd(np.sqrt(((pos[i,:,:] - pos[j,:,:])**2).sum(0)))
                
                D2[i,j] = np.nanmean(np.sqrt(((col[i,:,:] - col[j,:,:])**2).sum(0)))
                C2[i,j] = np.nanstd(np.sqrt(((col[i,:,:] - col[j,:,:])**2).sum(0)))
                
        atlas = {'bodypart':bodypart,
                  'C1': C1,
                  'D1': D1,
                  'C2': C2,
                  'D2': D2,
                  'names': N}
        
        return atlas
    
    @staticmethod
    def image_atlas(images,params,scales,shift=0,max_shape=None):
        moved = [[]]*len(images)
        if max_shape is None:
            max_shape = [max([int(float(im.shape[d])*1.1) for im in images]) for d in range(3)]
        
        dst_data = np.zeros(max_shape)

        for i in range(len(images)):
            print('Image number: ' + str(i))
            rotation = params['beta'][:3,:3,i]
            translation = params['beta0'][:,:3,i]
            
            iform = {}
            iform['rotation'] = np.linalg.inv(rotation)
            iform['translation'] = -((translation+shift)@iform['rotation']).T
            
            flow = Atlas.affine_flow(iform, dst_data, scale=scales[i])
            moved[i] = Atlas.image_warp(images[i],flow)
            
        
        sum_image = np.zeros(max_shape + [3])
        for im in moved:
            shp = [((max_shape[d]-im.shape[d])//2,max_shape[d]-im.shape[d]-(max_shape[d]-im.shape[d])//2) for d in range(3)]
            for c in range(3):
                sum_image[:,:,:,c] += np.pad(im[:,:,:,c], shp)

        return sum_image,moved


    def affine_flow(tform, grid, scale=1):
        """Converting an affine transformations into a vector flow for image 
            straightening
        
        Args: TODO
        
        Returns: TODO
        
        """
        
        if len(grid.shape) == 3:
            stacked_grid = (np.array(np.where(np.ones(grid.shape[0:3]))).T*scale)[:,[0,1,2]]
        else:
            stacked_grid = grid
        
        flow = (stacked_grid@tform['rotation'] + tform['translation'].T)/scale
            
        if len(grid.shape) == 3:
            flow = np.reshape(flow,(grid.shape[0],grid.shape[1],grid.shape[2],3))
            
        return flow
    
    def image_warp(im, flow):
        """Warping multi-channel volumetric input image using the 3D flow
        
        Args: TODO
        
        Returns: TODO
        
        """
        flow = np.floor(flow).astype(int)
        for d in range(flow.shape[3]):
            flow[:,:,:,d] = np.clip(flow[:,:,:,d], 0, im.shape[d]-1)
        
        mapped = im[flow[:,:,:,0],flow[:,:,:,1],flow[:,:,:,2],:]
        return mapped
    
    def cross_validate_(i,ims,neurons):
        ims_train = [ims[j] for j in range(len(ims)) if j != i]
        ims_test  = [ims[i]]

        annotations = [x.get_annotations() for x in ims_test]
        N = list(set([item for sublist in annotations for item in sublist])) 
        neurons_ = list(set(neurons).intersection(set(N)))
        atlas, aligned, _, params,_ = Atlas.train_atlas(ims_train, ims_train[0].bodypart,neurons_)
        Atlas.min_counts = -1
        N,col,pos,_ = Atlas.sort_mu(ims_test,neurons=atlas['names'])
        Atlas.min_counts = 2
        params,aligned,cost = Atlas.update_beta(np.hstack((pos,col)),atlas)
        return cost
        
    def cross_validate(ims,neurons=None,parallel=False,n_proc=10):
        if parallel:
            with Pool(n_proc) as p:
                cost = np.array(list(p.map(partial(Atlas.cross_validate_,ims=ims,neurons=neurons),np.arange(len(ims)))))
        else:
            cost =np.array(list(map(partial(Atlas.cross_validate_,ims=ims,neurons=neurons), np.arange(len(ims)))))

        return cost
    
    def optimal_label_positioning(mu,lambda_1=5,lambda_2=5,tol=1e0):
        # spring parameter (these were selected for the head of the worm, may need to play around with it for more dense plots)
        # minimum separation parameter 
         
        # Kamada-kawai loss function
        n = mu.shape[0] # the 2D neuron mean positions
        D = sp.spatial.distance.squareform(sp.spatial.distance.pdist(mu))
        ll = np.random.rand(n,1) # some random springyness to make the plot look "organic"
        ll = lambda_1*ll*ll.T + lambda_2
        L = np.vstack((
            np.hstack((D,D+ll-np.diag(np.diag(D+ll))+np.diag(ll))),
            np.hstack((D+ll-np.diag(np.diag(D+ll))+np.diag(ll),D-np.diag(np.diag(D+ll))+2*ll))
            ))
        K = 1/(L**2)
        k = 2*K
        l = L
        
        myfun = lambda x: Atlas.kk_cost(mu,x.reshape(mu.shape),k,l)

        # optimization        
        res = minimize(myfun, mu.reshape(-1), method='L-BFGS-B', tol=tol)
        
        # output annotation coordinates for each mu
        coor = res.x.reshape(mu.shape)# output annotation position
        return coor
    
    def kk_cost(mu,coor,k,l):
    # kamada-kawai force directed graph cost function
        y = np.vstack((mu,coor))
        
        pdist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(y))
        cost = np.triu(k*(pdist-l)**2,1).sum()
        
        return cost
    
