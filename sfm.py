import numpy as np
import matplotlib.pyplot as plt
from numba import njit, NumbaPerformanceWarning
import cv2
import os
from numba.typed import List

from plot_helpers import plotCoordinateFrame, set_axes_equal
from jacobian import jac
from manifold import K_from_vec, vec_from_K, SO3_from_vec, SE3_from_vec, vec_from_SE3, skew
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
        
        self.Ts = []
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
            self.Ts.append(np.eye(4))
            self.landmarks_new = landmarks_im
        else:
            # Next, we connect images together
            # TODO: Fix so you can check with multiple images at same time
            E, repeated_lm, new_lm = self._register_image(self.num_cam, landmarks_im)

            # # Get an estimate for this frame & new landmarks
            if repeated_lm == 0:
                T, Ps = self._pose_from_E(E, self.num_cam-1, self.num_cam)
            else:
                T, Ps, bad_idx = self._pose_from_pts(new_lm, self.num_cam)
                # self.seen = np.delete(self.seen, bad_idx, 0)

            # Update everything
            self.Ts.append(T)
            self.Ps = np.append(self.Ps, Ps, 0)
            assert self.Ps.shape[0] == self.num_lm

            # Run a small optimization with this new information

        self.num_cam += 1
        
    @property
    def _zs(self):
        # Organize measurements a  bit better
        return self.measurements[:,0].astype('int'), self.measurements[:,1].astype('int'), self.measurements[:,2:4]
    
    def _pose_from_E(self, E, im1_idx, im2_idx):
        # recover pose
        mm_idx1 = np.where(self.measurements[:,1]==im1_idx)[0]
        mm_idx2 = np.where(self.measurements[:,1]==im2_idx)[0]
        kp1 = self.measurements[ mm_idx1, 2:4 ]
        kp2 = self.measurements[ mm_idx2, 2:4 ]
        
        points, R, t, inliers = cv2.recoverPose(E, kp1, kp2, self.K[:3,:3])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.flatten()

        # Remove outliers
        inliers = inliers.flatten() != 0
        print(self.measurements)
        self.measurements = np.delete(self.measurements, mm_idx1[~inliers])
        self.measurements = np.delete(self.measurements, mm_idx2[~inliers])
        print(self.measurements)
        print("\t", kp1.shape[0] - np.sum(inliers), "are listed as outliers when recovering pose from E")
    
        # Triangulate points
        Ps_new = cv2.triangulatePoints(self.K@self.Ts[im1_idx], self.K@T, kp1.T, kp2.T).T
        Ps_new /= Ps_new[:,3:]
        
        return T, Ps_new[:,:3]
        
    def _pose_from_pts(self, new_lm, this_idx):
        # Convert kp idx -> landmark index -> 3d point
        mm_idx = np.where(self.measurements[:,1]==this_idx)[0]
        mm_idx_new = mm_idx[-new_lm:]
        mm_idx_old = mm_idx[:-new_lm]
        kps_old = self.measurements[mm_idx_old,2:4]
        lms_old = self.measurements[mm_idx_old,0].astype('int')
        
        # Use the estimating point estimates to find relative pose
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.Ps[lms_old], kps_old, self.K[:3,:3], None, reprojectionError=3, confidence=.99)
        inliers = inliers.flatten()
        T = np.eye(4)
        T[:3,:3] = SO3_from_vec(rvec.flatten())
        T[:3,3] = tvec.flatten()
        print("\t", kps_old.shape[0] - inliers.shape[0], "are listed as outliers when running PnP")
        
        # Triangulate all the other ones
        lms_new = self.measurements[mm_idx_new,0].astype('int')
        kp1_new = self.measurements[mm_idx_new,2:4] 
        kp2_new = self.measurements[mm_idx_new-1,2:4] 

        # TODO: If switching to multiple camera matches, this'll need to be a for loop
        Ps = cv2.triangulatePoints(self.K@self.Ts[this_idx-1], self.K@T, kp1_new.T, kp2_new.T)
        Ps = from_homogen(Ps.T)

        bad_idx = lms_new[np.where(Ps[:,2] < 0)[0]]

        return T, Ps, bad_idx
            
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
        E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, self.K[:3,:3], method=cv2.RANSAC, threshold=1.5, prob=0.99)
        inlier = inlier.flatten().astype('bool')
        match_idx = match_idx[inlier]  
        print("\t", match_idx.shape[0], "After Essential matrix RANSAC")

        # If it's the first iteration
        if self.num_lm == 0:
            n = match_idx.shape[0]
            lm = np.arange(n)

            # Add matches to measurements
            self.measurements = np.vstack((landmarks_all[match_idx[:,0]], landmarks_im[match_idx[:,1]]))
            self.measurements[:n,0] = lm
            self.measurements[n:,0] = lm
            a = np.lexsort((self.measurements[:,1], self.measurements[:,0]))
            self.measurements = self.measurements[a]

            # Add nonmatches to landmarks_new
            self.landmarks_new = np.delete(self.landmarks_new, match_idx[:,0], axis=0)
            not_seen = np.delete(landmarks_im, match_idx[:,1], axis=0)
            self.landmarks_new = np.vstack((self.landmarks_new, not_seen))

            return E, None, 
            
        else:
            # Get new landmarks (new ones have landmark value of NaN)
            new_lm = np.isnan( landmarks_all[match_idx[:,0],0] )
            old_lm = ~new_lm

            # Make array with old landmarks
            n_old = np.sum(old_lm)
            old_lm_numbers = landmarks_all[match_idx[old_lm,0],0]
            old_lm_matches = landmarks_im[match_idx[old_lm,1]]
            old_lm_matches[:,0] = old_lm_numbers
            print("\t", n_old, "Existing landmarks")

            # Make array with new landmarks
            n_new = np.sum(new_lm)
            lm = np.arange(self.num_lm, self.num_lm + n_new)
            new_lm_matches = np.vstack(((landmarks_all[match_idx[new_lm,0]], landmarks_im[match_idx[new_lm,1]])))
            new_lm_matches[:n_new,0] = lm
            new_lm_matches[n_new:,0] = lm
            print("\t", n_new, "New landmarks")

            # Put all togehter & sort
            self.measurements = np.vstack((self.measurements, old_lm_matches, new_lm_matches))
            a = np.lexsort((self.measurements[:,1], self.measurements[:,0]))
            self.measurements = self.measurements[a]

            # Delete landmarks from landmark new we just saw, and add in the ones from this image
            not_seen_all = np.delete(landmarks_all, match_idx[:,0], axis=0) # remove matches
            not_seen_all = not_seen_all[ np.isnan(not_seen_all[:,0]) ] # keep only kp that don't have landmarks
            not_seen_im = np.delete(landmarks_im, match_idx[:,1], axis=0) # remove all matches from last image
            self.landmarks_new = np.vstack((not_seen_all, not_seen_im))
            
            return E, n_old, n_new
        
if __name__ == "__main__":
    K_init = np.array([[3271.7198,    0.,     1539.6885, 0],
             [   0.,     3279.7956, 2027.496, 0],
             [   0.,        0.,        1.,0    ]])
    scale_percent = 25 # percent of original size
    K_init[:2] *= scale_percent/100

    folder = "data/statue"
    valid_imgs = [".jpg", ".png"]
    images = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in valid_imgs]

    sfm = StructureFromMotion(K_init, connections=1)

    for i in images[:2]:
        im1 = read_image(i, scale_percent)
        sfm.add_image(im1)
        print(i)

    # plot results
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sfm.Ps[:,0], sfm.Ps[:,1], sfm.Ps[:,2], s=1)
    for T in sfm.Ts:
        plotCoordinateFrame(T, ax=ax, k="--", size=1)
    set_axes_equal(ax)
    ax.set_zlabel("Z")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.view_init(-45, 0)
    plt.show()