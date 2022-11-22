import numpy as np
import matplotlib.pyplot as plt
from numba import njit, NumbaPerformanceWarning
import cv2
import os
from numba.typed import List

from plot_helpers import plotCoordinateFrame, set_axes_equal
from jacobian import jac
from manifold import SO3_from_vec
from optimize import levenberg_marquardt

import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
np.set_printoptions(suppress=True, precision=3) #, edgeitems=30, linewidth=100000)

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

class StructureFromMotion():
    def __init__(self, K, connections=1):
        # landmark index, camera index, kp, des
        self.num_cam = 0
        self.measurements = np.zeros((0,1+1+2+128), dtype='float32')
        self.landmarks_new = np.zeros((0,1+1+2+128), dtype='float32')
        self.connections = connections
        
        self.feat = cv2.SIFT_create()
        # Enforce unique matching
        self.match = cv2.BFMatcher(crossCheck=True)
        
        self.Ts = np.zeros((0,4,4))
        self.K = K
        self.Ps = np.zeros((0,3))

    @property
    def num_lm(self):
        if self.measurements.shape[0] == 0:
            return 0
        else:
            return int(self.measurements[-1,0])+1

    @property
    def landmarks(self):
        """
        Removes all duplicate rows for matching purposes. Assumes measurements are sorted by landmarks, then by most recent camera
        """
        if self.num_lm == 0:
            return self.measurements
        else:
            idx = np.zeros(self.num_lm, dtype=int)
            count = 0
            for i, val in enumerate(self.measurements[:,0]):
                if val > count:
                    idx[count] = i-1
                    count += 1

            return self.measurements[idx]
        
    def add_image(self, im):
        # First, we find keypoints
        kp, des = self.feat.detectAndCompute(im,None)
        kp = np.array([k.pt for k in kp])
        
        landmarks_im = np.hstack([
                            np.full((kp.shape[0], 1), np.NaN),
                            np.full((kp.shape[0], 1), self.num_cam), 
                            kp, 
                            des
                        ]).astype('float32')
    
        if self.num_cam == 0:
            self.Ts = np.eye(4)[None,:,:]
            self.landmarks_new = landmarks_im
        else:
            # Next, we connect images together
            # TODO: Fix so you can check with multiple images at same time
            E, repeated_lm, new_lm = self._register_image(self.num_cam, landmarks_im)

            # # Get an estimate for this frame & new landmarks
            if repeated_lm is None:
                T, Ps = self._pose_from_E(E, *new_lm)
            else:
                T, Ps = self._pose_from_pts(repeated_lm, *new_lm)

            # Update everything
            self.Ts = np.append(self.Ts, T[None,:,:], 0)
            self.Ps = np.append(self.Ps, Ps, 0)
            assert self.Ps.shape[0] == self.num_lm

        self.num_cam += 1

    def plot(self):
        # plot results
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.Ps[:,0], self.Ps[:,1], self.Ps[:,2], s=1)
        for T in self.Ts:
            plotCoordinateFrame(T, ax=ax, k="--", size=1)
        set_axes_equal(ax)
        ax.set_zlabel("Z")
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        # ax.view_init(-45, 0)
        plt.show()
        
    @property
    def _zs(self):
        # Sort by camera
        args = np.lexsort((self.measurements[:,0], self.measurements[:,1]))
        mm_sorted = self.measurements[args,2:4]
        lm_sorted = self.measurements[args,0]
        cam_sorted = self.measurements[args,1]

        # Get where to split everything
        idx = np.where(np.roll(cam_sorted,1)!=cam_sorted)[0]
        idx = np.append(idx, lm_sorted.shape[0])

        # Organize measurements a  bit better
        results = List((lm_sorted[idx[i]:idx[i+1]].astype('int'), mm_sorted[idx[i]:idx[i+1]]) for i in range(len(idx)-1))
        return results

    def optimize(self, tol=1e-4):
        self.K, self.Ts, self.Ps = levenberg_marquardt(
                                                residuals, 
                                                self.K, 
                                                self.Ts, 
                                                self.Ps, 
                                                self._zs, 
                                                jac, 
                                                max_iters=50, 
                                                lam=0.01, 
                                                lam_multiplier=4, 
                                                tol=tol, 
                                                verbose=10
                                            )

    def _add_measurements(self, new_mm):
        # Put all togehter & sort
        self.measurements = np.vstack((self.measurements, new_mm))
        a = np.lexsort((self.measurements[:,1], self.measurements[:,0]))
        self.measurements = self.measurements[a]
    
    def _pose_from_E(self, E, mm1, mm2):
        # Extract information we need fro measurements
        im1_idx = int(mm1[0,1])
        kp1 = mm1[:,2:4]
        kp2 = mm2[:,2:4]

        # recover pose
        points, R, t, inliers = cv2.recoverPose(E, kp1, kp2, self.K[:3,:3])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.flatten()

        # Put good measurements into measurement matrix
        inliers = inliers.flatten() != 0
        n = np.sum(inliers)
        lm = np.arange(n)
        mm_good = np.vstack((mm1[inliers], mm2[inliers]))
        mm_good[:n,0] = lm
        mm_good[n:,0] = lm
        self._add_measurements(mm_good) 

        # Remove outliers, insert into unmatched array
        bad_mm = np.vstack((mm1[~inliers], mm2[~inliers]))
        self.landmarks_new = np.vstack((self.landmarks_new, bad_mm))
        print("\t", np.sum(inliers), "were kept when recovering pose from E")
    
        # Triangulate points
        Ps_new = cv2.triangulatePoints(self.K@self.Ts[im1_idx], self.K@T, kp1[inliers].T, kp2[inliers].T).T
        Ps_new /= Ps_new[:,3:]
        
        return T, Ps_new[:,:3]
        
    def _pose_from_pts(self, old_lm, new_lm1, new_lm2):
        # Convert kp idx -> landmark index -> 3d point
        kps_old = old_lm[:,2:4]
        lms_old = old_lm[:,0].astype('int')
        
        # Use the estimating point estimates to find relative pose
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.Ps[lms_old], kps_old, self.K[:3,:3], None, reprojectionError=3, confidence=.99)
        inliers = inliers.flatten()
        T = np.eye(4)
        T[:3,:3] = SO3_from_vec(rvec.flatten())
        T[:3,3] = tvec.flatten()
        print("\t", inliers.shape[0], "are kept when running PnP")

        # Put everything where it needs to go
        self._add_measurements(old_lm[inliers])
        bad_mm = np.delete(old_lm, inliers, axis=0)
        bad_mm[:,0] = np.NaN
        self.landmarks_new = np.vstack((self.landmarks_new, bad_mm))


        # Triangulate all the other ones
        im1_idx = int(new_lm1[0,1])
        Ps = cv2.triangulatePoints(self.K@self.Ts[im1_idx], self.K@T, new_lm1[:,2:4].T, new_lm2[:,2:4].T)
        Ps = from_homogen(Ps.T)

        inliers = np.logical_and(Ps[:,2] > 0, np.linalg.norm(Ps, axis=1) < 50)
        n = np.sum(inliers)
        lm = np.arange(self.num_lm, self.num_lm+n)
        mm_good = np.vstack((new_lm1[inliers], new_lm2[inliers]))
        mm_good[:n,0] = lm
        mm_good[n:,0] = lm
        self._add_measurements(mm_good) 

        # Remove outliers, insert into unmatched array
        bad_mm = np.vstack((new_lm1[~inliers], new_lm2[~inliers]))
        self.landmarks_new = np.vstack((self.landmarks_new, bad_mm))
        print("\t", np.sum(inliers), "were kept when triangulating")

        return T, Ps[inliers]
            
    def _register_image(self, cam_idx, landmarks_im):
        print(f"Connecting Camera {cam_idx}")
        
        # Sort through seen landmarks
        landmarks_all = np.vstack((self.landmarks, self.landmarks_new))

        # Only use from last camera
        landmarks_all = landmarks_all[ landmarks_all[:,1] == cam_idx-1 ]

        # BFMatcher with default params
        knn_matches = self.match.match(landmarks_all[:,4:], landmarks_im[:,4:])
        knn_matches = sorted(knn_matches, key = lambda x : x.distance)

        # Get matches
        match_idx = np.array([[m.queryIdx, m.trainIdx] for m in knn_matches])
        kp1_match = landmarks_all[match_idx[:,0], 2:4]
        kp2_match = landmarks_im[match_idx[:,1], 2:4]
        print("\t", match_idx.shape[0], "Starting # matches")

        # Ransac & find essential matrix
        E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, self.K[:3,:3], method=cv2.RANSAC, threshold=1.5, prob=0.999)
        inlier = inlier.flatten().astype('bool')
        match_idx = match_idx[inlier]  
        print("\t", match_idx.shape[0], "After Essential matrix RANSAC")

        # If it's the first iteration
        if self.num_lm == 0:
            new_lm_mm = (landmarks_all[match_idx[:,0]], landmarks_im[match_idx[:,1]])

            # Add nonmatches to landmarks_new
            self.landmarks_new = np.delete(self.landmarks_new, match_idx[:,0], axis=0)
            not_seen = np.delete(landmarks_im, match_idx[:,1], axis=0)
            self.landmarks_new = np.vstack((self.landmarks_new, not_seen))

            return E, None, new_lm_mm
            
        else:
            # Get new landmarks (new ones have landmark value of NaN)
            new_lm = np.isnan( landmarks_all[match_idx[:,0],0] )
            old_lm = ~new_lm

            # Make array with old landmarks
            old_lm_matches = landmarks_im[match_idx[old_lm,1]]
            old_lm_matches[:,0] = landmarks_all[match_idx[old_lm,0],0]
            print("\t", np.sum(old_lm), "Existing landmarks")

            # Make array with new landmarks
            new_lm_matches = (landmarks_all[match_idx[new_lm,0]], landmarks_im[match_idx[new_lm,1]])
            print("\t", np.sum(new_lm), "New landmarks")

            # Delete landmarks from landmark new we just saw, and add in the ones from this image
            not_seen_all = np.delete(landmarks_all, match_idx[:,0], axis=0) # remove matches
            not_seen_all = not_seen_all[ np.isnan(not_seen_all[:,0]) ] # keep only kp that don't have landmarks
            not_seen_im = np.delete(landmarks_im, match_idx[:,1], axis=0) # remove all matches from last image
            self.landmarks_new = np.vstack((not_seen_all, not_seen_im))
            
            return E, old_lm_matches, new_lm_matches
        
if __name__ == "__main__":
    K_init = np.array([[3271.7198,    0.,     1539.6885, 0],
             [   0.,     3279.7956, 2027.496, 0],
             [   0.,        0.,        1.,0    ]])
    scale_percent = 5 # percent of original size
    K_init[:2] *= scale_percent/100

    folder = "data/statue"
    valid_imgs = [".jpg", ".png"]
    images = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in valid_imgs]

    sfm = StructureFromMotion(K_init, connections=1)

    for i in images[:2]:
        im1 = read_image(i, scale_percent)
        sfm.add_image(im1)
        if sfm.num_cam > 1:
            sfm.optimize(1)
            sfm.plot()