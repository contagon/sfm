import numpy as np
from numba import njit, NumbaPerformanceWarning
from numba.typed import List
from .manifold import skew
import scipy

# Since we don't use contiguous arrays for all these, numba gives annoying warnings.
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# projection function
@njit
def pi(K, T, P):
    return K@T@np.append(P, 1)

#------ Seperate partials to use in chain rule -----------#
@njit
def g_pi(p):
    return np.array([[1/p[2], 0, -p[0]/p[2]**2],
                   [0, 1/p[2], -p[1]/p[2]**2]])

@njit
def pi_K(K, T, P):
    Pp = T@np.append(P,1)
    return np.array([[Pp[0],   0.0, Pp[2],   0.0],
                     [  0.0, Pp[1],   0.0, Pp[2]],
                     [  0.0,   0.0,   0.0,   0.0]])
    
@njit
def pi_T(K, T, P):
    R = T[:3,:3]
    mat = np.hstack(( -R@skew(P), R ))
    return K[:3,:3]@mat

@njit
def pi_P(K, T, P):
    return K[:,:3]@T[:3,:3]

#------ Applying chain rule to seperate functions -----------#
@njit
def h_K(K, T, P):
    return g_pi(pi(K, T, P))@pi_K(K, T, P)

@njit
def h_T(K, T, P):
    return g_pi(pi(K, T, P))@pi_T(K, T, P)

@njit
def h_P(K, T, P):
    return g_pi(pi(K, T, P))@pi_P(K, T, P)


#-------- Helper to return triplet to make sparse matrix -----------#
@njit
def _block_sparse(As, indices):
    assert len(As) == len(indices)
    
    row = []
    col = []
    data = []
    for A, (start,stop) in zip(As, indices):
        m,n = A.shape
        for i in range(m):
            for j in range(n):
                if np.abs(A[i,j]) > 1e-4:
                    row.append(start+i)
                    col.append(stop+j)
                    data.append(A[i,j])
                                
    return data, (row,col)

#-------------  Construct the full Jacobian --------------#

@njit
def _jac_wrapped(K, Ts, Ps, zs):
    # Set things up!
    matrices = list()
    indices = list()
    M = Ts.shape[0]
    # N = Ps.shape[0]
    # Z = sum([cam_meas[0].shape[0] for cam_meas in zs])*2
    
    # Iterate through, saving where everything needs to go
    meas_idx = 0
    for cam_idx, (pt_indices, meassss) in enumerate(zs):
        for pt_idx, meas in zip(pt_indices, meassss):
            # indexing helpers
            o_cam = 4+(cam_idx-1)*6
            o_lm = 4+6*(M-1)+3*pt_idx

            # intrinsics
            matrices.append( h_K(K, Ts[cam_idx], Ps[pt_idx]) )
            indices.append( (2*meas_idx, 0) )
            
            # camera pose (but not first camera)
            if cam_idx != 0:
                matrices.append( h_T(K, Ts[cam_idx], Ps[pt_idx]) )
                indices.append( (2*meas_idx, o_cam) )
                
            # landmark position
            matrices.append( h_P(K, Ts[cam_idx], Ps[pt_idx]) )
            indices.append(  (2*meas_idx, o_lm) )
            
            meas_idx += 1

    return _block_sparse(matrices, indices)
    

def jac(K, Ts, Ps, zs):    
    M = Ts.shape[0]
    N = Ps.shape[0]
    Z = sum([cam_meas[0].shape[0] for cam_meas in zs])*2
        
    mat = scipy.sparse.coo_array( _jac_wrapped(K, Ts, Ps, zs), shape=(Z, 4 + 6*(M-1) + 3*N) )
            
    return mat.tocsc()




# Used previously when testing
def nder(f, x, eps=1e-6):
    fx = f(x)
    N = x.shape[0]
    M = fx.shape[0]
    d = np.zeros((M,N))
    for i in range(N):
        temp = x.copy()
        temp[i] += eps
        d[:,i] = (f(temp) - fx) / eps
        
    return d