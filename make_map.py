import sys
import numpy as np
import matplotlib.pyplot as plt

# Images
sys.path.insert( 1, '/home/alexgarcia/torreylabtoolsPy3' )
import visualization.contour_makepic as makepic
import util.calc_hsml as calc_hsml

def make_map(pos, m, star_pos, star_m, sfr, hrho, zm9, rmax,
             res, O_index=4, H_index=0, EAGLE_rho=False,
             rhocutidx=None, which='',save=True):
    ''' Crude First Code Taken from another project
    
    !! Can safely ignore this function !!
    '''
    
    pixl   =  1.000E-01
    nmin   =  16 #min particles needed to be observationally equivalent
        
    dr     =  1.00E-01 #kpc
    pixa   =  pixl**2.000E+00
    sigcut =  1.000E+00
    rhocut =  1.300E-01
    mcut   =  1.000E+01**sigcut * (pixa*1.000E+06)
    
    # Create map
    pixlims = np.arange(-rmax, rmax + pixl, pixl)
    pix   = len(pixlims) - 1
    pixcs = pixlims[:-1] + (pixl / 2.000E+00)
    rs    = np.full((pix, pix), np.nan, dtype = float)
    for r in range(0, pix):
        for c in range(0, pix):
            rs[r,c] = np.sqrt(pixcs[r]**2.000E+00 + 
                              pixcs[c]**2.000E+00 )
    
    if save:
        hsml = calc_hsml.get_particle_hsml( pos[:,0], pos[:,1], pos[:,2], DesNgb=32  )

        n_pixels = 720
        massmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml, m,
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e4  )

        np.save( './data/%s_gasmap.npy' %which, massmap )
        
        hsml = calc_hsml.get_particle_hsml( pos[:,0], pos[:,1], pos[:,2], DesNgb=32  )

        n_pixels = 720
        sfrmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml, sfr,
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e4  )

        np.save( './data/%s_sfrmap.npy' %which, sfrmap )
        
        Omassmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml,
            np.multiply(m, zm9[:,O_index]),
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e10  )
        
        Hmassmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml,
            np.multiply(m, zm9[:,H_index]),
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e10  )
        
        OH_map = (Omassmap / Hmassmap) * (1.00 / 16.00)
        OH_map = np.log10(OH_map) + 12
        
        np.save( './data/%s_metallicitymap.npy' %which, OH_map )
        
        hsml = calc_hsml.get_particle_hsml( star_pos[:,0], star_pos[:,1], star_pos[:,2], DesNgb=32  )

        n_pixels = 720
        starmassmap,image = makepic.contour_makepic( star_pos[:,0], star_pos[:,1], star_pos[:,2],
            hsml, star_m,
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e4  )

        np.save( './data/%s_starmap.npy' %which, starmassmap )
        
    else:
        massmap = np.load( './data/%s_gasmap.npy' %which )
        sfrmap  = np.load( './data/%s_sfrmap.npy' %which )
        OH_map  = np.load( './data/%s_metallicitymap.npy' %which )
        starmassmap = np.load( './data/%s_starmap.npy' %which )
        
    rhoidx = hrho > rhocut
    xym, x, y = np.histogram2d(pos[:,0], pos[:,1], weights = m, bins = [pixlims, pixlims])
    # Oxygen map
    xyo, x, y = np.histogram2d(pos[rhoidx,0], pos[rhoidx,1], weights = np.multiply(m[rhoidx], zm9[rhoidx,O_index]), bins = [pixlims, pixlims])
    # Hydrogen map
    xyh, x, y = np.histogram2d(pos[rhoidx,0], pos[rhoidx,1], weights = np.multiply(m[rhoidx], zm9[rhoidx,H_index]), bins = [pixlims, pixlims])
    
    xym       = np.transpose(xym)
    xyo       = np.transpose(xyo)
    xyh       = np.transpose(xyh)
    rs        = np.ravel( rs)
    xym       = np.ravel(xym)
    xyo       = np.ravel(xyo)
    xyh       = np.ravel(xyh)
    xyh[xyh < 1.000E-12] = np.nan
    
    xyoh     = xyo / xyh
    cutidx   =(xym > mcut) & (~np.isnan(xyoh)) & (xyoh > 0)
    _xyoh_   = np.where( cutidx, xym, np.nan )
    rs       =   rs[cutidx]
    xyoh     = xyoh[cutidx]
    xyoh     = np.log10(xyoh * (1.000E+00 / 1.600E+01)) + 1.200E+01
    rsortidx = np.argsort(rs)
    rs       =   rs[rsortidx]
    xyoh     = xyoh[rsortidx]
        
    return massmap, sfrmap, OH_map, starmassmap

def make_map_mass_sfr(pos, weight, rmax,
                      set_maxden=1.0e10,
                      set_dynrng=1.0e4 ):
    
    hsml = calc_hsml.get_particle_hsml( pos[:,0], pos[:,1], pos[:,2], DesNgb=32  )

    n_pixels = 720
    general_map, image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml, weight,
        xlen = rmax,
        pixels = n_pixels, set_aspect_ratio = 1.0,
        set_maxden = set_maxden, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
        set_dynrng = set_dynrng  )
    
    return general_map
    
def make_map_metallicity(pos, m, zm9, rmax,
                         set_maxden=1.0e10,
                         set_dynrng=1.0e4 ):
    O_index = 4
    H_index = 0
    
    hsml = calc_hsml.get_particle_hsml( pos[:,0], pos[:,1], pos[:,2], DesNgb=32  )

    n_pixels = 720
    Omassmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml,
            np.multiply(m, zm9[:,O_index]),
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = set_maxden, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = set_dynrng  )
        
    Hmassmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml,
        np.multiply(m, zm9[:,H_index]),
        xlen = rmax,
        pixels = n_pixels, set_aspect_ratio = 1.0,
        set_maxden = set_maxden, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
        set_dynrng = set_dynrng  )

    OH_map = (Omassmap / Hmassmap) * (1.00 / 16.00)
    OH_map = np.log10(OH_map) + 12
    
    return OH_map


if __name__ == "__main__":
    print("Hello World!")