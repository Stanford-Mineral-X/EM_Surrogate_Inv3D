## MD_flsification(d_var, d_obs, plt_OrNot, Q_quantile)
## Author: David Yin 
## Contact: yinzhen@stanford.edu
## Date: April 29, 2019


from sklearn.covariance import MinCovDet as MCD
from sklearn.covariance import EmpiricalCovariance as EC
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def RobustMD_flsification(num_sample, d_var, d_obs, prior_name, plt_OrNot, Q_quantile):
    
    '''
    This function falsifies the prior using Robust Mahalanobis Distance RMD.  
    d_var: the data variable, (nXp)
    d_obs: the data observation variable, (1xp)
    prior_name: name of the prior model for falsification, string
    plt_OrNot: True or False, to create the distribution plot of the calculated RMDs. 
    Q_quantile：the Q_quantile of the RMD distribution, 95 or 97.5 is suggested
    example: MD_flsification(d_pri, d_obs, True, 95) will produce the RMD_obs, RMD_pri, RMD_Q95, and plot them. 
    '''
    # mcd = EC().fit(d_var)
    mcd = MCD(random_state=0).fit(d_var)
    new_obs = d_obs-mcd.location_
    md_obs= np.sqrt(new_obs.dot(np.linalg.inv(mcd.covariance_)).dot(new_obs.T))
    print('Robust Mahalanobis Distance of d_obs = ', md_obs[0,0].round(decimals = 3))
    md_samples=[]
    for i in range(len(d_var)):
        sample = d_var[i:i+1, :]-mcd.location_
        md_samp = np.sqrt(sample.dot(np.linalg.inv(mcd.covariance_)).dot(sample.T))[0,0]
        md_samples.append(md_samp)
    md_samples = np.asarray(md_samples)
    print(str(Q_quantile)+'th Quantile of Robust Mahalanobis Distance is', \
          stats.scoreatpercentile(md_samples, Q_quantile).round(decimals=3))

    if plt_OrNot == True:
        plt.figure(figsize=(16,10))
        plt.scatter(np.arange(1,(len(d_var)+1)), md_samples, c=abs(md_samples), cmap ='winter_r', s=50, vmax = md_samples.max(), vmin=md_samples.min(),                    linewidths=1, edgecolor='k')
        plt.scatter([0], md_obs, c=md_obs,cmap ='winter_r', marker='D', s=110, vmax = md_samples.max(), vmin=md_samples.min(), linewidths=3, edgecolor='red')
        plt.text(5, md_obs,'$d_{obs}$',color='r',weight='bold',fontsize = 18, zorder = 100)
        plt.ylabel('Robust Mahalanobis distance')
        # plt.yscale('log')
        plt.xlabel('The number of realizations')
        plt.xlim(-0.05*num_sample, 1.05*num_sample)
        plt.hlines(y=stats.scoreatpercentile(md_samples, Q_quantile), xmin=-0.05*num_sample, xmax=num_sample*1.05, colors='red', linewidths=2, linestyles='--')
        cbar = plt.colorbar(fraction=0.035)
        cbar.ax.set_ylabel('RMD')
        # plt.title('(a)', loc='left')
        # plt.title('Prior falsification of "'+ prior_name+'" using Robust Mahalanobis Distance', \
        #           fontsize=18, loc='left', style='italic')
        # plt.savefig('fig_RMD.png', bbox_inches="tight", dpi=300)
    
    return md_obs[0,0].round(decimals = 3), stats.scoreatpercentile(md_samples, Q_quantile).round(decimals=3), md_samples