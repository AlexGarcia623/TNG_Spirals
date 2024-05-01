import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import illustris_python as il
import cmasher as cmr

from make_map import (
    make_map_mass_sfr,
    make_map_metallicity
)
from sfms import (
    calc_rsfr_io, calcrsfr, trans, center,
    calc_incl
)

best_of_best = [ # TNG50-1 spirals
    496788,
    502995,
    573738
]

run     = 'L35n2160TNG'
person  = 'zhemler'
out_dir = '/orange/paul.torrey/' + person + '/IllustrisTNG/' + run + '/output' + '/'

## Difference between out_dir and path...
## out_dir contains the particle data
## path includes the SubLink merger trees
## when doing this on Odyssey, combine into one!

path = '/orange/paul.torrey/IllustrisTNG/Runs/'
base_path = path + '/' + run
post_dir  = path + '/' + run + '/postprocessing'
tree_dir  = post_dir + '/trees/SubLink/'

xh = 7.600E-01
zo = 3.500E-01
mh = 1.6726219E-24

def get_tree_file(SUBHALO_ID, snap, tree_dir):
    file_located = False
    file_index   = None
    file_num     = 0 
    # Convert to raw SubhaloID
    SUBHALO_ID += int(snap * 1.00E+12)

    # Locate the SubLink tree file that the Subhalo exists in
    while not file_located:    
        tree_file = h5py.File( tree_dir + 'tree_extended.%s.hdf5' %file_num, 'r' )

        subID = np.array(tree_file.get('SubhaloIDRaw'))

        overlap = SUBHALO_ID in subID

        if (overlap):
            file_located = True
            file_index   = np.where( subID == SUBHALO_ID )[0][0]
        else:
            file_num += 1
            
    print('Found Subhalo ID %s in file %s' %(SUBHALO_ID, file_num))
    
    tree_file = h5py.File( tree_dir + 'tree_extended.%s.hdf5' %file_num, 'r' )
    
    return tree_file, file_index

desired_snaps = np.arange(58,100,1,dtype=int)[::-1] # Roughly z=0 -> z~0.7

for sub in best_of_best:
    print(f'Starting Subhalo {sub}')
    tree_file, file_index = get_tree_file(sub, 99, tree_dir)
    
    ## Get information about this galaxy's merger history
    subfind = tree_file["SubfindID"][file_index]
    assert(sub == subfind) ## Unnecessary, but I am a worrier
    rootID  = tree_file["SubhaloID"][file_index]
    fp      = tree_file["FirstProgenitorID"][file_index]
    snap    = tree_file["SnapNum"][file_index]
    
    hdr      = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    h        = hdr['HubbleParam']
    
    this_stellar_mass = tree_file['SubhaloMassType'][file_index, 4] * 1.00E+10 / h
    this_SFR          = tree_file['SubhaloSFR'][file_index]
    this_pos          = tree_file['SubhaloPos'][file_index]
    this_vel          = tree_file['SubhaloVel'][file_index]
    
    out_file = f'subhalo_{sub}_history.hdf5'
        
    while fp != -1 and snap in desired_snaps:
        # While there is a progenitor and we're above z~0.7 get and save info
        
        ## Load Header Info
        hdr  = il.groupcat.loadHeader(out_dir, snap)
        scf  = hdr['Time']
        z    = hdr['Redshift']
        
        ## Load Particle Info
        star_pos  = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Coordinates'      ])
        star_vel  = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Velocities'       ])
        star_mass = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Masses'           ])
        gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
        gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
        gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
        gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
        gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
        gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
        GFM_Metal = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])

        ## Manipulate galaxy
        gas_pos    = center(gas_pos, this_pos, box_size)
        star_pos   = center(star_pos, this_pos, box_size)
        gas_pos   *= (scf / h)
        star_pos  *= (scf / h)
        gas_vel   *= np.sqrt(scf)
        gas_vel   -= this_vel
        star_vel  *= np.sqrt(scf)
        star_vel  -= this_vel
        gas_mass  *= (1.000E+10 / h)
        star_mass *= (1.000E+10 / h)
        gas_rho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
        gas_rho   *= (1.989E+33    ) / (3.086E+21**3.00E+00)
        gas_rho   *= xh / mh

        ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
        ro2    = 2.000E+00 * ro

        riprime = ri + 0.25 * (ro - ri)

        sub_rsfr50 = calcrsfr(gas_pos, gas_sfr)

        sf_idx = gas_rho > 1.300E-01

        incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

        gas_pos  = trans(gas_pos, incl)
        star_pos = trans(star_pos, incl)
        gas_vel  = trans(gas_vel, incl)

        rmax = 50.0
        
        # Make maps and save them
        with h5py.File(out_file,'a') as f:
            current_snapshot = f.create_group(f'z={z:0.3f}')
            current_snapshot.create_dataset('Global SFR', data=this_SFR)
            current_snapshot.create_dataset('Global Stellar Mass', data=this_stellar_mass)
            
            gas_mass_map = make_map_mass_sfr(
                gas_pos, gas_mass, rmax,
                set_maxden = 1.0e10, set_dynrng = 1.0e4
            )
            current_snapshot.create_dataset('Gas Mass Map', data=gas_mass_map)
            
            star_mass_map = make_map_mass_sfr(
                star_pos, star_mass, rmax,
                set_maxden = 1.0e10, set_dynrng = 1.0e4
            )
            current_snapshot.create_dataset('Stellar Mass Map', data=star_mass_map)
            
            sfr_map = make_map_mass_sfr(
                gas_pos, gas_sfr, rmax,
                set_maxden = 1.0e-1, set_dynrng = 1.0e4
            )
            current_snapshot.create_dataset('SFR Map', data=sfr_map)
            
            metallicity_map = make_map_metallicity(
                gas_pos, gas_mass, GFM_Metal, rmax,
                set_maxden = 1.0e6, set_dynrng = 1.0e5
            )
            current_snapshot.create_dataset('Metallicity Map', data=metallicity_map)
            
        ## Get info for progenitor
        fpIndex = file_index + (fp - rootID)
    
        this_stellar_mass = tree_file['SubhaloMassType'][fpIndex, 4] * 1.00E+10 / h
        this_SFR          = tree_file['SubhaloSFR'][fpIndex]
        this_pos          = tree_file['SubhaloPos'][fpIndex]
        this_vel          = tree_file['SubhaloVel'][fpIndex]
    
        Sfid = tree_file["SubfindID"][fpIndex]
        fp   = tree_file["FirstProgenitorID"][fpIndex]
        snap = tree_file["SnapNum"][fpIndex]