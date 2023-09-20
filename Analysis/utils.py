import numpy as np
import pandas as pd
import scipy.optimize
import os

def maha_dist(data, mu, sigma):

    data_mu = data-mu
    inv_sigma = np.linalg.inv(sigma)
    left_data = np.dot(data_mu, inv_sigma)
    mahal = np.dot(left_data, data_mu.T)

    if type(mahal) ==np.float:
        return np.sqrt(mahal)
    else:
        return np.sqrt(mahal.diagonal())

def convert_coordinates(df, v1 = [[-40, 0, 0], [80, 0, -8]], v2=[[-40, 0, 0], [-40.8, 0, -12]], vRecenter=[0,0,0]):
    xyz = df[['X', 'Y', 'Z']]

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    assert np.isclose(0, np.dot(v1[1]-v1[0], v2[1]-v2[0])) , 'v1 v2 should be orthogonal'
    xyz = xyz - vRecenter
    v1_norm = (v1[1]-v1[0])/np.linalg.norm(v1)
    v2_norm = (v2[1]-v2[0])/np.linalg.norm(v2)
    v3_norm = np.cross(v1_norm, v2_norm)
    xnew = np.dot(xyz, v1_norm)
    znew = np.dot(xyz-v1[0], v2_norm)
    ynew = -np.dot(xyz-v1[0], v3_norm)
    h = -xnew
    r = np.sqrt(znew**2+ynew**2)
    th = np.arctan2(ynew, -znew)/np.pi*180.0

    # pack up and export
    df['xcyl'] = xnew
    df['ycyl'] = ynew
    df['zcyl'] = znew
    df['h'] = h
    df['r'] = r
    df['theta'] = th
    #df_atlas.to_csv(output_csv, float_format='%6g')

    return df

def covar_to_coord(covar):
    #covar should be 2x2 covariance matrix
    #compute eigenvalues and rotation theta
    
    a = covar[0][0]
    b = covar[1][0]
    c = covar[1][1]
    
    lam1 = (a+c)/2 + np.sqrt(((a-c)/2)**2+b**2)
    lam2 = (a+c)/2 - np.sqrt(((a-c)/2)**2+b**2)
    
    if (b==0):
        if a>=c:
            theta =0
        else:
            theta = np.pi/2
        
    else:
        theta = np.arctan2(lam1-a, b)
        
    return np.sqrt(lam1), np.sqrt(lam2), theta

def calc_costs(df_atlas, sigma, df_data):
    M = np.asarray(df_atlas[['X', 'Y', 'Z']])
    M_color = np.asarray(df_atlas[['R', 'G', 'B']])
    xyz_data = np.asarray(df_data[['X', 'Y', 'Z']])
    xyz_data = xyz_data[~np.isnan(xyz_data).any(axis=1)]
    rgb_data = np.asarray(df_data[['R', 'G', 'B']])
    rgb_data = rgb_data[~np.isnan(rgb_data).any(axis=1)]
    sigma_xyz = sigma[0:3,0:3, :]
    sigma_rgb = sigma[3:6, 3:6, :]
    log_like_xyz = np.zeros((M.shape[0], xyz_data.shape[0]))
    log_like_rgb = np.zeros((M_color.shape[0], rgb_data.shape[0]))
    Dxyz = np.zeros((xyz_data.shape[0], M.shape[0]))
    Drgb = np.zeros((rgb_data.shape[0], M_color.shape[0]))
    
    for i in range(M.shape[0]):

        Dxyz[:, i] = maha_dist(xyz_data, M[i,:], sigma_xyz[:,:,i])
        Drgb[:, i] = maha_dist(rgb_data, M_color[i,:], sigma_rgb[:,:,i])
        
    log_like_xyz = Dxyz.T #MxN
    log_like_rgb = Drgb.T
    
    row_xyz, col_xyz = scipy.optimize.linear_sum_assignment(log_like_xyz.T)
    row_rgb, col_rgb = scipy.optimize.linear_sum_assignment(log_like_rgb.T)
    
    cost_xyz = log_like_xyz.T[row_xyz, col_xyz].sum()/xyz_data.shape[0]
    cost_rgb = log_like_rgb.T[row_rgb, col_rgb].sum()/rgb_data.shape[0]
        
        
    return cost_xyz, cost_rgb

def check_accuracy(df):

    IDd = df.loc[pd.notna(df['ID']),:]
    correctID = df.loc[df['ID']== df['autoID_1']]
    correctSecond = df.loc[df['ID']== df['autoID_2']]

    correcttop2 = pd.concat([correctID,correctSecond]).drop_duplicates().reset_index(drop=True)

    per_ID = len(IDd.index)/len(df.index)
    per_correct = len(correctID.index)/len(IDd.index)
    per_top2 = len(correcttop2.index)/len(IDd.index)


    return per_ID, per_correct, per_top2, correctID, correcttop2

def get_cumul_acc(df):
    IDd = df.loc[pd.notna(df['ID']),:]
    corr1 = df.loc[df['ID']==df['autoID_1']]
    corr2 = df.loc[df['ID']==df['autoID_2']]
    corr3 = df.loc[df['ID']==df['autoID_3']]
    corr4 = df.loc[df['ID']==df['autoID_4']]
        
    corr_cum_2 = pd.concat([corr1,corr2]).drop_duplicates().reset_index(drop=True)
    corr_cum_3 = pd.concat([corr_cum_2,corr3]).drop_duplicates().reset_index(drop=True)
    corr_cum_4 = pd.concat([corr_cum_3,corr4]).drop_duplicates().reset_index(drop=True)

    per_ID = len(IDd.index)/len(df.index)
    per_corr_1 = len(corr1.index)/len(IDd.index)
    per_corr_2 = len(corr_cum_2.index)/len(IDd.index)
    per_corr_3 = len(corr_cum_3.index)/len(IDd.index)
    per_corr_4 = len(corr_cum_4.index)/len(IDd.index)

    return IDd, per_ID, [per_corr_1, per_corr_2, per_corr_3, per_corr_4]