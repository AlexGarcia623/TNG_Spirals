import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher

import illustris_python as il

mpl.rcParams['text.usetex']        = True
# mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.size']          = 20

fs_og = 24
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth']  = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size']  = 7.5
mpl.rcParams['ytick.major.size']  = 7.5
mpl.rcParams['xtick.minor.size']  = 3.5
mpl.rcParams['ytick.minor.size']  = 3.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

data_dir = './data/'

desired_galaxies = np.load(data_dir+'galaxies.npy')

for galaxy in desired_galaxies:
    
    gas_map = np.load(data_dir + f'{galaxy}_gasmap.npy')
    
    left = gas_map[:,:gas_map.shape[1]//2]
    
    right = gas_map[:,gas_map.shape[1]//2:]
    flip_mirror = right[::-1,::-1] # Flip and mirror
        
    # fig, axs = plt.subplots(1, 4, figsize=(10,8),
    #                         gridspec_kw={'width_ratios':[1,1,0.1,1]})
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    cm = 'cmr.dusk'
    cm2 = 'cmr.neutral'
    
    ax.imshow(gas_map, norm=LogNorm(), cmap=cm)
    
    # axs[0].imshow(left, norm=LogNorm(), cmap=cm)
    # axs[1].imshow(flip_mirror, norm=LogNorm(), cmap=cm)
    
    # difference = left - flip_mirror
    # difference /= np.max(gas_map)
    
#     c=axs[3].imshow(difference, norm=LogNorm(), cmap=cm2)
#     # plt.colorbar(c,fraction=0.1)

#     axs[0].text(0.5,1.007,r'Left',ha='center',transform=axs[0].transAxes)
#     axs[1].text(0.5,1.007,r'Right',ha='center',transform=axs[1].transAxes)
#     axs[3].text(0.5,1.007,r'Difference',ha='center',transform=axs[3].transAxes)
    
    # for ax in axs:
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    ax.axis('off')
    # axs[0].axis('off')
    # axs[1].axis('off')
    # axs[2].axis('off')
    
    plt.tight_layout()
    
    # plt.subplots_adjust(wspace=0.05)
    
    plt.savefig('./plots/'+ f'{galaxy}_structure.pdf',bbox_inches='tight')