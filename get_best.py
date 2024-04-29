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

out_file = 'TNG_spiral_sample.hdf5'

best_ids = [
    496788,	562029,	604516,
    502995,	567382,	611780,
    527839,	571908,	629266,
    536654,	573738,	635905,
    538370,	575863,	639432,
    545437,	584511,	642303,
    550934,	587500,
    552414,	601693,
]

xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

snap    = 99
run     = 'L35n2160TNG'
person  = 'zhemler'
out_dir  = '/orange/paul.torrey/' + person + '/IllustrisTNG/' + run + '/output' + '/'

hdr  = il.groupcat.loadHeader(out_dir, snap)
box_size = hdr['BoxSize']
scf      = hdr['Time']
h        = hdr['HubbleParam']
z        = hdr['Redshift']

fields  = ['SubhaloMassType','SubhaloPos','SubhaloVel','SubhaloSFR']
prt     = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)

prt['SubhaloMassType'] *= 1.00E+10 / h

for sub in best_ids:
    this_stellar_mass = prt['SubhaloMassType'][sub, 4]
    this_SFR          = prt['SubhaloSFR'][sub]
    this_pos          = prt['SubhaloPos'][sub]
    this_vel          = prt['SubhaloVel'][sub]

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

    with h5py.File(out_file,'a') as f:
        current_group = f.create_group(f'subhalo_{sub}')
        current_group.create_dataset('Global SFR', data=this_SFR)
        current_group.create_dataset('Global Stellar Mass', data=this_stellar_mass)

        gas_mass_map = make_map_mass_sfr(gas_pos, gas_mass, rmax,
                                         set_maxden=1.0e10,
                                         set_dynrng=1.0e4 )
        star_mass_map = make_map_mass_sfr(star_pos, star_mass, rmax,
                                          set_maxden=1.0e10,
                                          set_dynrng=1.0e4 )
        sfr_map = make_map_mass_sfr(gas_pos, gas_sfr, rmax,
                                    set_maxden=1.0e-1,
                                    set_dynrng=1.0e4 )
        metallicity_map = make_map_metallicity(gas_pos, gas_mass, GFM_Metal, rmax,
                                           set_maxden=1.0e6,
                                           set_dynrng=1.0e5)
        current_group.create_dataset('Gas Mass Map', data=gas_mass_map)
        current_group.create_dataset('Stellar Mass Map', data=star_mass_map)
        current_group.create_dataset('SFR Map', data=sfr_map)
        current_group.create_dataset('Metallicity Map', data=metallicity_map)