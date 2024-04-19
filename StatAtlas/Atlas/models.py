import copy

import scipy as sp
import numpy as np

import utils

from tqdm.auto import trange

# %%
class Atlas:
    """Class for training an atlas from multiple annotated images by aligning 
    the images using affine transformations
    """
    
    def __init__(
            self,
            min_counts=2,
            epsilon=[1e+3,1e+3]
        ):

        self.min_counts = min_counts
        self.epsilon = epsilon

    
    def update_beta(self,X,model):
        """Updating the transformation parameters for different images
    
        Args:
            X (np.ndarray): Nx(3+C) neural point clouds
            model (dict): Current atlas estimate
            
        Returns:
            params (dict): Regression params
            aligned (numpy.ndarray): Aligned point clouds
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
            R = utils.MCR_solver(
                np.concatenate((
                    X[idx,:3,j],
                    np.ones((idx.sum(),1))),
                    1
                ),
                model['mu'][idx,:3],
                model['sigma'][:3,:3,idx]
            )

            beta[:3,:3,j] = R[:3,:3]
            beta0[:,:3,j] = R[3,None]
            
            # solving for colors
            R = utils.MCR_solver(
                np.concatenate((
                    X[idx,3:,j], 
                    np.ones((idx.sum(),1))),
                    1
                ),
                model['mu'][idx,3:],
                model['sigma'][3:,3:,idx]
            )

            beta[3:,3:,j] = R[:C,:C]
            beta0[:,3:,j] = R[C,None]
            
            aligned[:,:,j] = (X[:,:,j])@beta[:,:,j]+beta0[:,:,j]
            
            # adding the cost of positions
            cost[0] += sum([
                    sp.spatial.distance.mahalanobis(
                        aligned[i,:3,j].squeeze(),
                        model['mu'][i,:3].squeeze(),
                        sigma_inv[:3,:3,i]
                    ) for i in np.where(idx)[0]
                ])
            
            # adding the cost of colors
            cost[1] += sum([
                sp.spatial.distance.mahalanobis(
                    aligned[i,3:,j].squeeze(),
                    model['mu'][i,3:].squeeze(),
                    sigma_inv[3:,3:,i])
                    for i in np.where(idx)[0]
                ])
        
        params['beta'] = beta
        params['beta0'] = beta0
        
        return params, aligned, cost
    

    def initialize_atlas(self,col,pos, match_indices, train_indices):
        """Initialize atlas by finding the best image for aligning all other 
        images
    
        Args:
            col (np.ndarray): NxC color array (C num channels)
            pos (np.ndarray): Nx3 position array
                
        Returns:
            atlas (dict): Initialized atlas
            X (np.ndarray): Initially aligned point cloud
        """
		
        # memory allocation
        atlas = {}
        cost = np.zeros((pos.shape[2],len(match_indices)))
        aligned = [np.zeros((
            pos.shape[0], 
            pos.shape[1]+col.shape[1],
            pos.shape[2]
        ))]*len(match_indices)
        
        # alignment of samples to best fit worm
        for i in range(pos.shape[2]):
            for j, num in enumerate(match_indices):
                S0,R0,T0 = utils.scaled_rotation(
                    pos[:,:,i],
                    pos[:,:,num]
                )
                cost[i,j] = np.sqrt(
                    np.nanmean(
                        np.nansum((pos[:,:,i]@(R0*S0)+T0-pos[:,:,num])**2,1),0
                    ))
                aligned[j][:,:3,i] = pos[:,:,i]@(R0*S0)+T0
                aligned[j][:,3:,i] = col[:,:,i]
        
        jidx = np.argmin(cost.sum(0))
        X = aligned[jidx]

        atlas['mu'] = np.nanmean(X[:,:,train_indices],2)          
        
        return atlas,X
    
    def estimate_mu(self,mu,aligned):
        """Estimate the neuron centers and colors using updated and aligned 
        images
    
        Args:
            mu (np.ndarray): Previous value of mu, size Nx(3+C)
            aligned (np.ndarray): Aligned point clouds, size Nx(3+C)xK
                
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
            mu (np.ndarray): Previous value of mu, size Nx(3+C)
            aligned (np.ndarray): Aligned point clouds, size Nx(3+C)xK
                
        Returns:
            sigma (dict): Updated value of covariance Nx(3+C)x(3+C)
        """
        
        # memory allocation
        sigma   = np.zeros((mu.shape[1],mu.shape[1],mu.shape[0]))
        
        # computing the covariances
        for i in range(aligned.shape[0]):
            sigma[:,:,i] = np.ma.cov(np.ma.masked_array(aligned[i,:,:]))
        
        # well-condition the sigmas by adding epsilon*identity
        sigma[:3,:3,:] = sigma[:3,:3,:] + reg[0]*np.eye(sigma[:3,:3,:].shape[0])[:,:,None]
        sigma[3:,3:,:] = sigma[3:,3:,:] + reg[1]*np.eye(sigma[3:,3:,:].shape[0])[:,:,None]
        
        for i in range(aligned.shape[0]):
            # decorrelate color and position 
            sigma[:3,3:,i] = 0
            sigma[3:,:3,i] = 0
            
            # diagonalize color covariances
            sigma[3:,3:,i] = np.diag(np.diag(sigma[3:,3:,i]))
        
        return sigma
    
    
    
    def sort_mu(self,ims,neurons=None):
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
        counts = (~np.isnan(pos.sum(1))).sum(1)

        # filtering the neurons based on min_count of the missing data
        good_indices = np.logical_and( counts>self.min_counts, 
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
        
    
    
    def train_atlas(
            self,
            ims,
            bodypart,
            neurons=None,
            n_iter=10,
            train_indices = [],
            match_indices = []
        ):
        """Main function for estimating the atlas of positions and colors
    
        Args:
            positions (np.ndarray): Nx3 position array
            colors (np.ndarray): NxC color array
            bodypart (str): Worm's body part (head or tail)
                
        Returns:
            atlas (dict): Trained atlas
            aligned_coord (np.ndarray): Aligned point clouds
        """

        if train_indices == []:
            train_indices = np.arange(len(ims))
        if match_indices == []:
            match_indices = np.arange(len(ims))

        train_indices = np.asarray(train_indices)
        match_indices = np.asarray(match_indices)
        
        ims = copy.deepcopy(ims)
        bodypart = ims[0].bodypart
        
        for i in range(len(ims)):
            im = ims[i]
            if neurons is not None:
                im.neurons = [
                    im.neurons[i] 
                    for i in range(len(im.neurons))
                    if im.neurons[i].annotation in neurons
                ]
                
        annotations = [x.get_annotations() for x in ims]
        colors = [x.get_colors_readout() for x in ims]
                              
        C = colors[0].shape[1]
        N,col,pos,counts = self.sort_mu(ims)
        
        # initialization
        model,aligned = self.initialize_atlas(col,pos, match_indices, train_indices)
        init_aligned = np.hstack((pos,col))
        
        cost = []

        pbar = trange(n_iter)
        pbar.set_description('Initializing ...')

        for iteration in pbar:
            # updating means.
            model['mu'] = self.estimate_mu(
                model['mu'],
                aligned[:,:,train_indices]
            )
            
            # updating sigma
            model['sigma'] = self.estimate_sigma(
                model['mu'],
                aligned[:,:,train_indices],reg=self.epsilon
            )
            
            # updating aligned
            params,aligned,cost_ = self.update_beta(
                init_aligned,
                model
            )

            cost.append(cost_)
            pbar.set_description(
                'Positon cost: {:.2f}, Color cost: {:.2f}'.format(
                    cost_[0],
                    cost_[1]
                )
            )
            
        model['mu'] = self.estimate_mu(model['mu'],aligned[:,:,train_indices])
        model['sigma'] = self.estimate_sigma(model['mu'],aligned[:,:,train_indices])
        
        # store the result for the output
        atlas = {
            'bodypart':bodypart,
            'mu': model['mu'],
            'sigma': model['sigma'],
            'names': N,
            'aligned': aligned
        }

        # store worm specific parameters inside their corresponding
        # class
        aligned_coord = []
        for j in range(len(annotations)):
            perm = np.array([
                N.index(x) if x in N else -1 
                for x in annotations[j]
            ])
            aligned_coord.append(
                np.array([
                    aligned[perm[n],:,j] 
                    if perm[n] != -1 
                    else np.zeros((C+3))*np.nan 
                    for n in range(len(perm)
                )]
            ))
            
        return atlas,aligned_coord,params,cost,counts
    
    
    def train_distance_atlas(self,annotations,scales,positions,colors,bodypart):
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
        good_indices = np.logical_and( counts>self.min_counts, 
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
                
        atlas = {
            'bodypart':bodypart,
            'C1': C1,
            'D1': D1,
            'C2': C2,
            'D2': D2,
            'names': N
        }
        
        return atlas
    
    
    def image_atlas(self,images,params,scales,shift=0,max_shape=None):
        moved = [[]]*len(images)
        if max_shape is None:
            max_shape = [max([
                int(float(im.shape[d])*1.1) 
                for im in images]) 
                for d in range(3)
            ]
        
        dst_data = np.zeros(max_shape)

        for i in range(len(images)):
            print('Image number: ' + str(i))
            rotation = params['beta'][:3,:3,i]
            translation = params['beta0'][:,:3,i]
            
            iform = {}
            iform['rotation'] = np.linalg.inv(rotation)
            iform['translation'] = -((translation+shift)@iform['rotation']).T
            
            flow = self.affine_flow(iform, dst_data, scale=scales[i])
            moved[i] = self.image_warp(images[i],flow)
            
        
        sum_image = np.zeros(max_shape + [3])
        for im in moved:
            shp = [((
                max_shape[d]-im.shape[d])//2,
                max_shape[d]-im.shape[d]-\
                    (max_shape[d]-im.shape[d])//2)
                for d in range(3)
            ]
            for c in range(3):
                sum_image[:,:,:,c] += np.pad(im[:,:,:,c], shp)

        return sum_image,moved


    def affine_flow(self,tform,grid,scale=1):
        """Converting an affine transformations into a vector flow for image 
        straightening
        """
        
        if len(grid.shape) == 3:
            stacked_grid = (np.array(np.where(np.ones(grid.shape[0:3]))).T*scale)[:,[0,1,2]]
        else:
            stacked_grid = grid
        
        flow = (stacked_grid@tform['rotation'] + tform['translation'].T)/scale
            
        if len(grid.shape) == 3:
            flow = np.reshape(flow,(grid.shape[0],grid.shape[1],grid.shape[2],3))
            
        return flow
    
    def image_warp(self,im,flow):
        """Warping multi-channel volumetric input image using the 3D flow
        """
        flow = np.floor(flow).astype(int)
        for d in range(flow.shape[3]):
            flow[:,:,:,d] = np.clip(flow[:,:,:,d], 0, im.shape[d]-1)
        
        mapped = im[flow[:,:,:,0],flow[:,:,:,1],flow[:,:,:,2],:]
        return mapped