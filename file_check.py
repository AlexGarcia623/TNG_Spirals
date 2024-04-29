# Standard Imports
import numpy as np
import matplotlib as mpl
mpl.use('agg') # Used for HPC... supresses plt.show
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

######### Might need to pip install these? #########
import h5py
import cmasher as cmr ## optional... used for cmap
####################################################

file = 'TNG_spiral_sample.hdf5'

make_plots = False ## Flag to make figures of each

best_ids = [ ## as identified by Qianhui/Alex
    496788,	562029,	604516,
    502995,	567382,	611780,
    527839,	571908,	629266,
    536654,	573738,	635905,
    538370,	575863,	639432,
    545437,	584511,	642303,
    550934,	587500,
    552414,	601693,
]

def to_value(dataset):
    '''h5py datasets are quirky with single values
    
    This returns a single value to a float
    '''
    return float(np.array(dataset))

for target in best_ids:
    with h5py.File(file,'r') as f:
        group = f[f'subhalo_{target}']

        stellar_mass = to_value(np.log10(group['Global Stellar Mass']))
        sfr = to_value(group['Global SFR'])
        
        t1 = 'log Stellar Mass'
        print(f"Subhalo: {target}")
        print(f"\tlog Stellar Mass: {stellar_mass:0.3f} log M_sun")
        print(f"\t{'SFR':>{len(t1)}}: {sfr:0.3f} M_sun / yr")
        
        ## Load Maps
        gas_mass_map  = group['Gas Mass Map'] ## Msun / kpc^2
        star_mass_map = group['Stellar Mass Map'] ## Msun / kpc^2
        sfr_map       = group['SFR Map'] ## Msun / yr / kpc^2
        metals_map    = group['Metallicity Map'] ## log O/H + 12

        ## Make plots
        if make_plots:
            all_maps = [
                gas_mass_map, star_mass_map, sfr_map, metals_map
            ]
            maps_names = [
                'gas_mass', 'star_mass', 'sfr', 'metals'
            ]
            for index, Map in enumerate(all_maps):
                plt.clf()

                fig = plt.figure(figsize=(8,8))
                ax = plt.gca()

                ax.imshow(Map, norm=LogNorm(), cmap='cmr.dusk')

                ax.axis('off')
                plt.tight_layout()

                plt.savefig('./best_plots/'+ f'{target}_{maps_names[index]}.pdf',bbox_inches='tight')
