import sys
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import illustris_python as il

from make_map import make_map
from sfms import (
    calc_rsfr_io, calcrsfr, trans, center,
    calc_incl
)

xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

def tng(snap):
    run     = 'L35n2160TNG'
    # person  = 'alexgarcia'
    person  = 'zhemler'
    out_dir  = '/orange/paul.torrey/' + person + '/IllustrisTNG/' + run + '/output' + '/'
    
    print('Loading Header')
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    h        = hdr['HubbleParam']
    z        = hdr['Redshift']
    
    print('Loading Subhalo and Halo Catalogs')
    fields  = ['SubhaloMassType','SubhaloPos','SubhaloVel','SubhaloSFR']
    prt     = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)
    subs    = il.groupcat.loadHalos(out_dir,snap,fields=['GroupFirstSub'])
    
    prt['SubhaloMassType'] *= 1.00E+10 / h
    
    min_sm = 10.0
    max_sm = 11.0
    min_sfr = 0.5
    max_sfr = 10.0
        
    desired_galaxies = subs[(prt['SubhaloMassType'][subs,4] > 1.000E+01**min_sm) & 
                            (prt['SubhaloMassType'][subs,4] < 1.000E+01**max_sm) &
                            (prt['SubhaloSFR'][subs] > min_sfr) & 
                            (prt['SubhaloSFR'][subs] < max_sfr)]
    
    for sub in desired_galaxies:
        print('Starting Galaxy: %s' %sub)
        plt.clf()
        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()
        
        this_stellar_mass = prt['SubhaloMassType'][ sub, 4 ]
        this_pos          = prt['SubhaloPos'][sub] #* (scf / h)
        this_vel          = prt['SubhaloVel'][sub] #* np.sqrt( scf )

        star_pos  = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Coordinates'      ])
        star_vel  = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Velocities'       ])
        star_mass = il.snapshot.loadSubhalo(out_dir, snap, sub, 4, fields = ['Masses'           ])
        gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
        gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
        gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
        gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
        gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
        gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
        ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
        XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]

        GFM_Metal = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])

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
        gas_rho   *= XH / mh

        OH = ZO/XH * 1.00/16.00

        Zgas = np.log10(OH) + 12

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
        res  = 2160
        massmap = make_map(gas_pos, gas_mass, star_pos, star_mass, gas_sfr, gas_rho, GFM_Metal,
                           rmax, res, save=True, which=sub)
    
#         mappable=ax.imshow( massmap, cmap=plt.cm.inferno, extent=[-rmax,rmax,-rmax,rmax],
#                             norm=LogNorm() )
    
#         # ax.axis('off')
#         plt.tight_layout()
#         plt.savefig(f'./plots/{sub}_star_map.pdf',bbox_inches='tight')
    
    
if __name__ == "__main__":
    tng(99)
