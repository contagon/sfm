import numpy as np
import scipy
from manifold import K_from_vec, vec_from_K, SE3_from_vec, vec_from_SE3

def levenberg_marquardt(residual, K_init, Ts_init, Ps_init, zs, jac, lam=10.0, lam_multiplier=10.0, max_iters=100, tol=1e-8, verbose=False, line_start=""):
    # Loosely adapted from https://github.com/jjhartmann/Levenberg-Marquardt-Algorithm/blob/master/LMA.py
    K = K_init.copy()
    Ts = Ts_init.copy()
    Ps = Ps_init.copy()
    M = Ts.shape[0]
    N = Ps.shape[0]
    
    prev_cost =np.sum( residual(K, Ts, Ps, zs)**2 )
    I = scipy.sparse.identity(4 + (Ts.shape[0]-1)*6 + Ps.shape[0]*3)
    lam_init = lam
    
    if verbose:
        print(f"{line_start}Starting cost: {prev_cost}")

    for k in range(max_iters):
        # Make right hand side
        J = jac(K, Ts, Ps, zs)
        JtJ = J.T@J
        
        # Make left hand side
        r = residual(K, Ts, Ps, zs)
        b = J.T@r
            
        # Run with this linearization
        cost = prev_cost + 1
        while cost > prev_cost:
            # Find our delta
            A = JtJ + lam*I
            delta = scipy.sparse.linalg.spsolve(A, b)
            
            # Make copy of current estimate
            Kstar = K.copy()
            Tstar = Ts.copy()
            Pstar = Ps.copy()
            
            # Apply updates
            Kstar = K_from_vec( vec_from_K(K) + delta[:4] )
            for i in range(1,M):
                Tstar[i] = Ts[i]@SE3_from_vec( delta[4+6*(i-1):4+6*i] )
            Pstar += delta[4+6*(M-1):].reshape((N,3))
            
            # See if it got us anywhere
            cost = np.sum( residual(Kstar, Tstar, Pstar, zs)**2 )
                        
            # If it didn't work, try a smaller step
            if cost > prev_cost:
                lam *= lam_multiplier
                
            if lam > 1e9:
                print("{line_start}LM failed to converge")
                return K, Ts, Ps
                        
        # If our step worked, increment and keep going
        K = Kstar
        Ts = Tstar
        Ps = Pstar
        
        if lam > lam_init:
            lam /= lam_multiplier
        
        # See if it was a small improvement, and if so, be done
        if np.abs(prev_cost - cost) < tol:
            break
        else:
            prev_cost = cost
            
        if verbose and (k+1) % verbose == 0:
            print(f"{line_start}{k+1}, \t Cost: {cost} \t Lam: {lam}")

    if verbose:
        print(f"{line_start}Ending cost: {cost}") 

    return K, Ts, Ps