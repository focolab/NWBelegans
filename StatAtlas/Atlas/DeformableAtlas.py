# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:55:03 2021

@author: Amin
"""
from scipy.spatial.transform import Rotation as R
from Methods.StatAtlas.Atlas.Atlas import Atlas
from Methods.Straightening import utils as U
from Methods.StatAtlas.Atlas import Helpers
from sklearn.decomposition import PCA
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import scipy as sp
import matplotlib
import copy

class Object: pass

# %%
class DeformableAtlas(Atlas):
    n_iter      = 20
    gamma       = 0 # 1e2
    eta         = 0 # .1
    reg_iter    = 10
    epsilon     = 1e+2
    
    @staticmethod
    def straighten(ims):
        for i in range(len(ims)):
            im = ims[i]
            centerline = U.centerline_gw(0,im.get_positions(im.scale)[:,:,None],radius=1,epsilon=.01,n_points=20)
            straight,_,_ = U.straighten_pp(0,centerline[:,:,None],centers=im.get_positions(im.scale)[:,:,None])
            for n in range(len(im.neurons)):
                im.neurons[n].position = straight[n,:]
            im.scale = 1
        
        return ims
    
    @staticmethod
    def piecewise_rigid(atlas,aligned):
        for i in range(aligned.shape[2]):
            U.estimate_piecewise_rigid(atlas['mu'][:,:3], aligned[:,:3])
        
    @staticmethod
    def train_straightened_atlas(ims,neurons=None):
        ims_straight = copy.deepcopy(ims)
        bodypart = ims[0].bodypart
        
        for i in range(len(ims_straight)):
            im = ims_straight[i]
            if neurons is not None:
                im.neurons = [im.neurons[i] for i in range(len(im.neurons)) if im.neurons[i].annotation in neurons]
        
        ims_straight = DeformableAtlas.straighten(ims_straight)
        atlas_straight, aligned_straight = Atlas.train_atlas(ims,bodypart)
            
        DeformableAtlas.visualize(atlas_straight,aligned_straight,title_str='Atlas Straight',save=True,file='atlas_straight')

        return atlas_straight,aligned_straight


    @staticmethod
    def train_deformable_atlas(ims_,neurons=None,tesselation=None):
        """Main function for estimating the atlas of positions and colors
    
        Args:
            ims (list):
                
        Returns:
            atlas (dict):
            aligned_coord (numpy.ndarray):
            
        """
        
        ims = copy.deepcopy(ims_)
        bodypart = ims[0].bodypart
        
        for i in range(len(ims)):
            im = ims[i]
            if neurons is not None:
                im.neurons = [im.neurons[i] for i in range(len(im.neurons)) if im.neurons[i].annotation in neurons]


        annotations = [x.get_annotations() for x in ims]
        colors = [x.get_colors_readout() for x in ims]
                  
        C = colors[0].shape[1]

        
        N,col,pos = Atlas.sort_mu(ims)
        
        # initialization
        model,aligned = Atlas.initialize_atlas(col,pos)
        # model,aligned = DeformableAtlas.initialize_atlas(col,pos,tesselation=tesselation)
        
        
        init_aligned = np.hstack((pos,col))
        
        cost = []
        
        for iteration in range(DeformableAtlas.n_iter):
            # updating means.
            model['mu'] = DeformableAtlas.estimate_mu(model['mu'],aligned)
            
            # updating sigma
            model['sigma'] = DeformableAtlas.estimate_sigma(model['mu'],aligned,reg=DeformableAtlas.epsilon)

            # DeformableAtlas.visualize_pretty(atlas,[],title_str='iter'+str(iteration),save=True,file='iter'+str(iteration))

            # updating aligned
            params,aligned,cost_ = DeformableAtlas.update_beta(init_aligned,model,tesselation=tesselation)
            cost.append(cost_)
            print('Iteration: ' + str(iteration) + ' - Cost (pos,col): ' + str(cost_))

        model['mu'] = Atlas.estimate_mu(model['mu'],aligned)
        model['sigma'] = Atlas.estimate_sigma(model['mu'],aligned)
        
        # store the result for the output
        atlas = {'bodypart':bodypart, \
                'mu': model['mu'], \
                'sigma': model['sigma'], \
                'names': N,
                'aligned': aligned}
            

        # store worm specific parameters inside their corresponding class
        aligned_coord = []
        for j in range(len(annotations)):
            perm = np.array([N.index(x) if x in N else -1 for x in annotations[j]])
            aligned_coord.append(np.array([aligned[perm[n],:,j] if perm[n] != -1 else np.zeros((C+3))*np.nan for n in range(len(perm))]))


        return atlas,aligned_coord,cost,params
    
    
    @staticmethod
    def update_beta(X,model,tesselation=None):
        """Updating the transformation parameters for different images
    
        Args:
            X (numpy.ndarray): 
            model (dict):
            
        Returns:
            params (dict):
            aligned (numpy.ndarray):
        """
        
        params = {}
        
        # computing beta for each training worm based using multiple
        # covariance regression solver
        
        tforms = [[]]*X.shape[2]
        iforms = [[]]*X.shape[2]
        tesses = [[]]*X.shape[2]
        itsses = [[]]*X.shape[2]
        
        aligned = np.zeros(X.shape)
        
        C = X.shape[1]-3
        
        beta    = np.zeros((C,C,X.shape[2]))
        bet0    = np.zeros((1,C,X.shape[2]))
        
        sigma_inv = np.array([np.linalg.inv(model['sigma'][:,:,i]) 
                  for i in range(model['sigma'].shape[2])]).transpose([1,2,0])

        cost = [0,0]
        for j in range(X.shape[2]):
            idx = ~np.isnan(X[:,:,j]).all(1)
            
            # solving for positions
            src = X[idx,:3,j].copy()
            dst = model['mu'][idx,:3].copy()
            
            tform,iform,tess,itss,cost_,icst = U.estimate_piecewise_rigid(src,dst,
                  gamma=DeformableAtlas.gamma,eta=DeformableAtlas.eta,n_iter=DeformableAtlas.reg_iter,
             tesselation=tesselation,inverse_tesselation=tesselation)
            
            straight = U.extended_procrustes_flow(tform, itss, X[:,:3,j])
            
            # DeformableAtlas.visualize_spr(src,dst,straight,cost_,icst)
            
            tforms[j] = tform
            iforms[j] = iform
            tesses[j] = tess
            itsses[j] = itss
            
            
            # solving for colors
            R = Helpers.MCR_solver(model['mu'][idx,3:], \
                np.concatenate((X[idx,3:,j], np.ones((idx.sum(),1)) ), 1), \
                model['sigma'][3:,3:,:])
            
            beta[:,:,j] = R[:C,:C]
            bet0[:,:,j] = -R[C,None]
            
            aligned[:,:3,j] = straight
            aligned[:,3:,j] = X[:,3:,j]@beta[:,:,j]+bet0[:,:,j]

            cost[0] += sum([sp.spatial.distance.mahalanobis(aligned[i,:3,j].squeeze(),model['mu'][i,:3].squeeze(),sigma_inv[:3,:3,j]) for i in np.where(idx)[0]])
            cost[1] += sum([sp.spatial.distance.mahalanobis(aligned[i,3:,j].squeeze(),model['mu'][i,3:].squeeze(),sigma_inv[3:,3:,j]) for i in np.where(idx)[0]])

            # cost[0] = cost[0] + np.sqrt(((aligned[idx,:3,j]-model['mu'][idx,:3])**2).sum(1)).sum()
            # cost[1] = cost[1] + np.sqrt(((aligned[idx,3:,j]-model['mu'][idx,3:])**2).sum(1)).sum()
            
        params['beta'] = beta
        params['bet0'] = bet0
        
        params['tforms'] = tforms
        params['iforms'] = iforms
        params['tesses'] = tesses
        params['itsses'] = itsses
        
        return params, aligned, cost
    
    @staticmethod
    def initialize_atlas(col,pos,tesselation=None):
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
        cost    = np.zeros((pos.shape[2],pos.shape[2]))*np.inf
        aligned = [np.zeros((pos.shape[0], pos.shape[1]+col.shape[1],pos.shape[2]))]*pos.shape[2]
        
        # alignment of samples to best fit worm
        for i in range(pos.shape[2]):
            for j in range(1):
                if i == j:
                    cost[i,j] = 0
                    aligned[j][:,:3,i] = pos[:,:,i]
                    aligned[j][:,3:,i] = col[:,:,i]
                    continue
                
                src = pos[:,:,i]
                dst = pos[:,:,j]
                
                tform,iform,tess,itss,cost_,icst_ = U.estimate_piecewise_rigid(src,dst,
                               gamma=DeformableAtlas.gamma,eta=DeformableAtlas.eta,n_iter=DeformableAtlas.reg_iter,
                               tesselation=tesselation,inverse_tesselation=tesselation)
                straight = U.extended_procrustes_flow(tform, itss, src)
                
                
                cost[i,j] = cost_[-1,:].sum()
                aligned[j][:,:3,i] = straight
                aligned[j][:,3:,i] = col[:,:,i]
                
        jidx = np.argmin(cost.sum(0))
        X = aligned[jidx]

        # initializations
        model['mu']     = np.nanmean(X,2)          
        
        return model,X
    
    @staticmethod
    def image_atlas(images,params,scales,max_shape=None,smoothing=1):
        moved = [[]]*len(images)
        
        if max_shape is None:
            max_shape = [max([int(float(im.shape[d])*1.1) for im in images]) for d in range(3)]
            
        dst_data = np.zeros(max_shape)
        
        for i in range(len(images)):
            print('Image number: ' + str(i))
            
            iform = params['iforms'][i]
            # itss  = params['itsses'][i]
            tess  = params['tesses'][i]
            
            flow = U.extended_procrustes_flow(iform, tess, dst_data, scale=scales[i])
            
            for d in range(3):
                flow[:,:,:,d] = sp.ndimage.gaussian_filter(flow[:,:,:,d],smoothing)


            moved[i] = U.image_warp(images[i],flow)
        
        sum_image = np.zeros(max_shape + [3])
        for im in moved:
            shp = [((max_shape[d]-im.shape[d])//2,max_shape[d]-im.shape[d]-(max_shape[d]-im.shape[d])//2) for d in range(3)]
            for c in range(3):
                sum_image[:,:,:,c] += np.pad(im[:,:,:,c], shp)

        return sum_image,moved
    
    def major_axis_align(atlas,aligned,params,shift=0):
        pca = PCA(n_components=3)
        pca.fit(atlas['mu'][:,:3])
        projection = pca.components_.T
        projection = projection[:,[1,0,2]]
        
        shift = (-(atlas['mu'][:,:3]@projection).min(0)+shift)
        
        iprojection = np.linalg.inv(projection)
        ishift = -(shift@iprojection)
        
        mus = atlas['mu'][:,:3]@projection + shift
        cov = np.zeros((3,3,mus.shape[0]))
        for i in range(cov.shape[2]):
            cov[:,:,i] = projection.T@atlas['sigma'][:3,:3,i]@projection
            
        
        proj_atlas = copy.deepcopy(atlas)
        proj_atlas['mu'][:,:3] = mus
        proj_atlas['sigma'][:3,:3,:] = cov
        
        proj_params = copy.deepcopy(params)
        
        for i in range(len(proj_params['tforms'])):
            for j in range(len(proj_params['tforms'][i])):
                proj_params['tforms'][i][j]['translation'] = proj_params['tforms'][i][j]['translation']@projection + shift
                proj_params['tforms'][i][j]['rotation'] = proj_params['tforms'][i][j]['rotation']@projection
        
        for i in range(len(proj_params['iforms'])):
            for j in range(len(proj_params['iforms'][i])):
                proj_params['iforms'][i][j]['translation'] = proj_params['iforms'][i][j]['translation'] + ishift@proj_params['iforms'][i][j]['rotation']
                proj_params['iforms'][i][j]['rotation'] = iprojection@proj_params['iforms'][i][j]['rotation']


        proj_aligned = copy.deepcopy(aligned)
        for i in range(len(aligned)):
            proj_aligned[i][:,:3] = aligned[i][:,:3]@projection+shift
            
        for i in range(len(proj_params['tforms'])):
            proj_params['tesses'][i].points = proj_atlas['mu'][:,:3].copy()
            proj_params['itsses'][i].points = proj_aligned[i][:,:3].copy()
        
        return proj_atlas,proj_aligned,proj_params
    
    def visualize_tes(tesselation,names=None,tform=None,fontsize=15,save=False,file=None):
        if names is None:
            names = ['']*len(tesselation.vertices)
        plt.figure(figsize=(15,5))
        com = np.array([tesselation.points[tri].mean(0) for tri in tesselation.vertices])
        colors = plt.cm.hsv(np.linspace(0,1,len(tesselation.vertices)+1)[0:-1])[:,0:3]
        
        for i in range(len(tesselation.neighbors)):
            for j in range(len(tesselation.neighbors[i])):
                if tesselation.neighbors[i][j] < 0 or i < j:
                    continue
                src = tesselation.points[tesselation.vertices[i],:2].mean(0)
                dst = tesselation.points[tesselation.vertices[j],:2].mean(0)
                plt.plot([src[1],dst[1]], [src[0],dst[0]], 'b', markersize=0, linestyle='--')
        
        for i in range(com.shape[0]):
            plt.scatter(com[i,1],com[i,0],c=colors[i,:],label=names[i],marker='*',s=200,edgecolors='k')
            plt.scatter(tesselation.points[tesselation.vertices[i],1],
                        tesselation.points[tesselation.vertices[i],0],
                        c=colors[i,:][None,:])
            
        if tform is not None:
            for i in range(com.shape[0]):
                angles = R.from_matrix(tform[i]['rotation']).as_euler('yxz', degrees=True)
                arc = matplotlib.patches.Arc(com[i,[1,0]],6,6,angle=0,theta1=0,theta2=angles[0])
                plt.gca().add_patch(arc)
                
                plt.quiver(com[i,1],com[i,0],tform[i]['translation'][1],tform[i]['translation'][0],
                           edgecolor='k',facecolor='w',linewidth=1,angles='xy')

        plt.axis('equal')
        plt.grid('on')
        plt.legend(fontsize=fontsize)
        plt.title('Tesselation',fontsize=fontsize)
        
        if save:
            plt.savefig(file+'.png',format='png')
            plt.savefig(file+'.pdf',format='pdf')
            plt.close('all')
        else:
            plt.show()
    
    def visualize_spr(src,dst,straight,cost,icst,fontsize=15,linewidth=3,save=False,file=None):
        plt.figure(figsize=(15,10))
        plt.subplot(311)
        plt.scatter(src[:,1],src[:,0],c='r',label='Src')
        plt.scatter(dst[:,1],dst[:,0],c='b',label='Dst')
        plt.scatter(straight[:,1],straight[:,0],c='k',label='Moved')
        plt.legend(fontsize=fontsize)
        plt.grid('on')
        
        normalize = lambda x: (x-x.min())/(x.max()-x.min())
        
        plt.subplot(312)
        plt.plot(normalize(cost[:,0]),label='Reconstruction',linewidth=linewidth)
        plt.plot(normalize(cost[:,1]),label='Regularization',linewidth=linewidth)
        plt.plot(normalize(cost.sum(1)),label='Sum',linewidth=linewidth)
        plt.grid('on')
        plt.legend(fontsize=fontsize)
        plt.ylabel('Forward Cost',fontsize=fontsize)
        
        plt.subplot(313)
        plt.plot(normalize(icst[:,0]),label='Reconstruction',linewidth=linewidth)
        plt.plot(normalize(icst[:,1]),label='Regularization',linewidth=linewidth)
        plt.plot(normalize(icst.sum(1)),label='Sum',linewidth=linewidth)
        plt.grid('on')
        plt.legend(fontsize=fontsize)
        plt.ylabel('Inverse Cost',fontsize=fontsize)
        plt.xlabel('Iterations',fontsize=fontsize)
    
    
        if save:
            plt.savefig(file+'.png',format='png')
            plt.savefig(file+'.pdf',format='pdf')
            plt.close('all')
        else:
            plt.show()
           
    
    
    def cross_validate_(i,ims,neurons,tesselation):
        ims_train = [ims[j] for j in range(len(ims)) if j != i]
        ims_test  = [ims[i]]

        annotations = [x.get_annotations() for x in ims_test]
        N = list(set([item for sublist in annotations for item in sublist])) 
        neurons_ = list(set(neurons).intersection(set(N)))

        atlas, aligned, _, params = DeformableAtlas.train_deformable_atlas(ims_train,neurons_,tesselation=tesselation)
        Atlas.min_counts = -1
        N,col,pos = DeformableAtlas.sort_mu(ims_test,neurons=atlas['names'])
        Atlas.min_counts = 2
        params,aligned,cost = DeformableAtlas.update_beta(np.hstack((pos,col)),atlas,tesselation=tesselation)
        return cost
    
    
    def cross_validate(ims,neurons=None,tesselation=None,parallel=False,n_proc=10):
        if parallel:
            with Pool(n_proc) as p:
                cost = np.array(list(p.map(partial(DeformableAtlas.cross_validate_,ims=ims,neurons=neurons,tesselation=tesselation),np.arange(len(ims)))))
        else:
            cost =np.array(list(map(partial(DeformableAtlas.cross_validate_,ims=ims,neurons=neurons,tesselation=tesselation), np.arange(len(ims)))))
            
        return cost
