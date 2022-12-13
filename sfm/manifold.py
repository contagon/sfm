import numpy as np
from numba import njit

#----- Helpers for handling manifolds & matrix representations --------#
@njit
def skew(u):
    return np.array([[   0, -u[2],  u[1]],
                    [ u[2],     0, -u[0]],
                    [-u[1],  u[0],     0]])

@njit
def K_from_vec(x):
    K = np.zeros((3,4))
    K[0,0] = x[0]
    K[1,1] = x[1]
    K[2,2] = 1
    K[0:2,2] = x[2:4]
    return K

@njit
def vec_from_K(K):
    return np.array([K[0,0], K[1,1], K[0,2], K[1,2]])

@njit
def SO3_from_vec(w):
    wx = skew(w)
    theta = np.linalg.norm(w)
    if np.abs(theta) < 0.0001:
        R = np.eye(3) + wx + wx@wx/2 + wx@wx@wx/6
    
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        R = np.eye(3) + A*wx + B*wx@wx
    
    return R

@njit
def SE3_from_vec(u):
    # https://www.ethaneade.com/lie.pdf
    w = u[:3]
    p = u[3:]
    wx = skew(w)
    theta = np.linalg.norm(w)
    if np.abs(theta) < 0.0001:
        R = np.eye(3) + wx + wx@wx/2 + wx@wx@wx/6
        V = np.eye(3) + wx/2 + wx@wx/6 + wx@wx@wx/24
    
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        C = (1 - A) / theta**2

        R = np.eye(3) + A*wx + B*wx@wx
        V = np.eye(3) + B*wx + C*wx@wx

    T = np.hstack((R, (V@p).reshape((3,1))))
    T = np.vstack(( T, np.array([[0.0,0,0,1]]) ))
    return T
 
@njit
def vec_from_SE3(T):
    # https://www.ethaneade.com/lie.pdf
    xi = np.zeros(6)
    
    # Log on the rotations
    R = T[:3,:3]
    p = T[:3,3].copy()
    theta = np.arccos( (np.trace(R) - 1) / 2 )
    x = (R - R.T)*theta / (2*np.sin(theta))
    
    xi[0] = R[2,1] - R[1,2]
    xi[1] = R[0,2] - R[2,0]
    xi[2] = R[1,0] - R[0,1]
    
    if theta != 0:
        xi[:3] *= theta / (2*np.sin(theta))
        
    # And on the translation
    wx = skew(xi[:3])
    if theta < .0001:
        V = np.eye(3) + wx/2 + wx@wx/6 + wx@wx@wx/24
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta**2
        C = (1 - A) / theta**2
        V = np.eye(3) + B*wx + C*wx@wx
        
    Vinv = np.linalg.inv(V)
    xi[3:] = Vinv@p
    return xi

