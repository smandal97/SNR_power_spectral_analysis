#!/usr/bin/env python
# coding: utf-8



import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import variance
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

def centeredDistanceMatrix(n):
    # make sure n is odd
    x,y = np.meshgrid(range(n),range(n))
    return np.sqrt((x-(n/2)+1)**2+(y-(n/2)+1)**2)


def interpolate_function(d,x,y):
    f  = interp1d(x, y)
    im = f(d.flat).reshape(d.shape).T
    return im


def make_sphere(center, arrsize, radius):
    distance = np.linalg.norm(np.subtract(np.indices(arrsize).T, center), axis=-1)
    mask = np.ones(arrsize) * (distance < radius)
    return mask


def get_radial_profile(dat, cen, nr=80):
    center  = np.array(dat.shape) * cen
    rs = np.linalg.norm(np.subtract(np.indices(dat.shape).T, center), axis=-1)
    #dr = rs.max()/float(nr)
    r_indices = rs.astype(int) #np.floor(rs/dr).astype(int)
    f = np.zeros(nr)
    fhist,rhist = np.histogram(r_indices,bins=nr,weights=dat)
    fden,rden = np.histogram(r_indices,bins=nr)
    rhist[-1] += 1. #rhist = rhist[:-1] #0.5*(rhist[1:] + rhist[:-1])
    fhist_true = np.divide(fhist, fden, out=np.zeros_like(fhist), where=fden!=0)
    fhist_true = np.pad(fhist_true, (0, 1), 'edge')
    return rs, rhist, fhist_true


def get_radial_profile_alt(dat, cen, nr=20):
    nx = dat.shape[0]
    center = np.array(dat.shape) * cen
    rs = np.linalg.norm(np.subtract(np.indices(dat.shape).T, center), axis=-1)
    dr = rs.max()/float(nr)
    r_interv = np.linspace(0.,rs.max(),nr)
    f1 = np.zeros(nr)
    f2 = np.zeros(nr)
    for i in range(nx):
        for j in range(nx):
            k = int(np.floor(rs[i,j]/dr))
            if(k>=0 and k<nr):
                f1[k] += dat[i,j]
                f2[k] += 1.
    f_prof = f1/f2
    return rs, r_interv, f_prof


def get_radial_avg_img(dat,cen):
    rs,r,f = get_radial_profile_alt(dat,cen)
    #n = dat.shape[0]
    #d = centeredDistanceMatrix(n)
    avg_img = interpolate_function(rs,r,f).T
    return avg_img
    

def get_delta(sig,I,M,eps):
    sig1    = sig/np.sqrt(1.+eps)
    sig2    = sig*np.sqrt(1.+eps)
    im_c_g1 = gaussian_filter(I*M, sigma=sig1, mode='wrap')
    im_c_g2 = gaussian_filter(I*M, sigma=sig2, mode='wrap')
    m_c_g1  = gaussian_filter(M, sigma=sig1, mode='wrap')
    m_c_g2  = gaussian_filter(M, sigma=sig2, mode='wrap')
    term1   = np.divide(im_c_g1, m_c_g1, out=np.zeros_like(im_c_g1), where=m_c_g1!=0)
    term2   = np.divide(im_c_g2, m_c_g2, out=np.zeros_like(im_c_g2), where=m_c_g2!=0)
    S       = (term1-term2) * M
    return S


def get_variance(I,M,r,num=60,sigma_min=1e0,sigma_max=5e2):
    sigmas = np.geomspace(sigma_min,sigma_max,num)
    Ps     = np.zeros(num)
    eps    = 1e-3
    for i in range(num):
        sigma = sigmas[i]
        S     = get_delta(sigma,I,M,eps)
        Ps[i] = variance(S)
    return sigmas, Ps


def polymask(im, pt_x, pt_y):
    width, height = im.shape
    polygon=[(pt_x[0]*width, pt_y[0]*height), (pt_x[1]*width, pt_y[1]*height),
             (pt_x[2]*width, pt_y[2]*height), (pt_x[3]*width, pt_y[3]*height)]
    poly_path=Path(polygon)
    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
    mask = poly_path.contains_points(coors)
    return mask.reshape(height, width)




fits_image_filename = "tycho_snr.fits"
with fits.open(fits_image_filename) as hdul:
    data = hdul[0].data

mask_a = polymask(data, np.array([.647,.185,.206,.691]), np.array([.058,.891,.90,.055]))
mask_b = polymask(data, np.array([.147,.945,.915,.120]), np.array([.196,.642,.675,.228]))
radius = 450. #in pixels
mask_c = 1. - make_sphere(np.array(data.shape)/2., data.shape, radius)
mask_c = mask_c.astype(bool)
data   = data.astype(float)
mask   = 1. - (mask_a+mask_b+mask_c)

rad_avg = get_radial_avg_img(data, 0.5)



# get_variance() computes the delta variance as a function of length scale sigma.
# To convert to a power spectrum, multiply by sigma**2 (for 2D images)
# nsig is the array of sigma values, V_nomask here is for Tycho without mask, V_mask is with mask,
# and V_mr is with mask and with the radial average subtracted off.

nsig,V_nomask = get_variance(data, np.ones_like(data), data.shape[0]/2.)
nsig,V_mask   = get_variance(data , mask, data.shape[0]/2.)
nsig,V_mr     = get_variance(np.divide(data, rad_avg, out=np.zeros_like(data), where=rad_avg!=0) , mask, data.shape[0]/2.)





