import matplotlib.pyplot as plt
import cv2
import numpy as np
from numba import njit

@njit
def to_homogen(p):
    return np.hstack(( p, np.ones((p.shape[0], 1)) ))

@njit
def from_homogen(p):
    p /= p[:,-1:]
    return p[:,:-1]

def read_image(filename, scale):
    # load images
    im = cv2.imread(filename)

    width = int(im.shape[1] * scale / 100)
    height = int(im.shape[0] * scale / 100)
    dim = (width, height)

    # resize image
    im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    return im

@njit
def h(K, T, P):
    p_homo = K@T@to_homogen(P).T
    return from_homogen(p_homo.T)

@njit
def residuals(K, Ts, Ps, zs):     
    # Compute expected measurements
    steps = [0] + [cam_meas[0].shape[0] for cam_meas in zs]
    steps = np.cumsum(np.array(steps))
    res = np.zeros(2*steps[-1])

    for cam_idx, (pt_idx, meas) in enumerate(zs):
        p_prime = h(K, Ts[cam_idx], Ps[pt_idx])
        res[2*steps[cam_idx]:2*steps[cam_idx+1]] = (meas - p_prime).flatten()
    
    return res


def plotMatches(im1, im2, locs1, locs2, filename=None, figsize=(6,4)):
    fig = plt.figure(figsize=figsize)
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    
    if len(im1.shape) >= 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) >= 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        
    im[0:im1.shape[0], 0:im1.shape[1]] = im1
    im[0:im2.shape[0], im1.shape[1]:] = im2
    
    plt.imshow(im, cmap='gray')
    for ((x1, y1), (x2, y2)) in zip(locs1, locs2):
        plt.plot([x1, x2+im1.shape[1]], [y1,y2], 'r', lw=0.5)

    plt.show()

