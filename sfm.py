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
        self.landmarks_seen = np.zeros((0,1+1+2+128), dtype='float32')
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
        if self.landmarks_seen.shape[0] == 0:
            return 0
        else:
            return int(self.landmarks_seen[-1,0])+1

    @property
    def landmarks(self):
        """
        Removes all duplicate rows for matching purposes. Assumes landmarks_seen are sorted by landmarks, then by most recent camera
        """
        if self.num_lm == 0:
            return self.landmarks_seen
        else:
            idx = np.zeros(self.num_lm, dtype=int)
            count = 0
            for i, val in enumerate(self.landmarks_seen[:,0]):
                if val > count:
                    idx[count] = i-1
                    count += 1

            return self.landmarks_seen[idx]
        
    def add_image(self, im):
        # First, we find keypoints
        kp, des = self.feat.detectAndCompute(im,None)
        kp = np.array([k.pt for k in kp])
        
        im_landmarks = np.hstack([
                            np.full((kp.shape[0], 1), np.NaN),
                            np.full((kp.shape[0], 1), self.num_cam), 
                            kp, 
                            des
                        ]).astype('float32')
    
        if self.num_cam == 0:
            self.Ts.append(np.eye(4))
            self.landmarks_new = im_landmarks
        else:
            # Next, we connect images together
            # TODO: Fix so you can check with multiple images at same time
            E, repeated_lm, new_lm = self._register_image(self.num_cam, im_landmarks)

            # # Get an estimate for this frame & new landmarks
            # if repeated_lm == 0:
            #     T, Ps = self._pose_from_E(E, new_lm, this_idx-1, this_idx)
            # else:
            #     T, Ps, bad_idx = self._pose_from_pts(new_lm, this_idx)
            #     self.seen = np.delete(self.seen, bad_idx, 0)

            # # Update everything
            # self.Ts.append(T)
            # self.Ps = np.append(self.Ps, Ps, 0)
            # assert self.Ps.shape[0] == self.seen.shape[0]

            # Run a small optimization with this new information

        self.num_cam += 1
        
    @property
    def _zs(self):
        # Organize measurements a  bit better
        return self.landmarks_seen[:,0].astype('int'), self.landmarks_seen[:,1].astype('int'), self.landmarks_seen[:,2:4]
    
    def _pose_from_E(self, E, new_lm, im1_idx, im2_idx):
        # recover pose
        kp1 = self.kp[im1_idx][ self.seen[-new_lm:,im1_idx] ]
        kp2 = self.kp[im2_idx][ self.seen[-new_lm:,im2_idx] ]
        
        points, R, t, inliers = cv2.recoverPose(E, kp1, kp2, self.K[:3,:3])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.flatten()
        inliers = inliers.flatten() != 0
        print("\t", kp1.shape[0] - np.sum(inliers), "are listed as outliers when recovering pose from E")
    
        # Triangulate points
        Ps_new = cv2.triangulatePoints(self.K@self.Ts[im1_idx], self.K@T, kp1.T, kp2.T).T
        Ps_new /= Ps_new[:,3:]
        
        return T, Ps_new[:,:3]
        
    def _pose_from_pts(self, new_lm, this_idx):
        # Convert kp idx -> landmark index -> 3d point
        lm_idx = np.nonzero(self.seen[:,this_idx])[0]
        lm_old = lm_idx[:-new_lm]
        lm_new = lm_idx[-new_lm:]
        kps = self.kp[this_idx][ self.seen[lm_old,this_idx] ]
        
        # Use the estimating point estimates to find relative pose
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.Ps[lm_old], kps, self.K[:3,:3], None, reprojectionError=3, confidence=.99)
        inliers = inliers.flatten()
        T = np.eye(4)
        T[:3,:3] = SO3_from_vec(rvec.flatten())
        T[:3,3] = tvec.flatten()
        print("\t", kps.shape[0] - inliers.shape[0], "are listed as outliers when running PnP")
        
        # Triangulate all the other ones
        Ps = []
        bad_idx = []
        for lm in lm_new:
            cam_saw = np.nonzero(self.seen[lm])[0]
            assert len(cam_saw) > 1
            assert cam_saw[-1] == this_idx
            idx1 = cam_saw[0]
            Ps_new = cv2.triangulatePoints(self.K@self.Ts[idx1], self.K@T, self.kp[idx1][self.seen[lm, idx1]], self.kp[this_idx][self.seen[lm, this_idx]])
            Ps_new = Ps_new.flatten() / Ps_new[3]
            if Ps_new[2] >= 0 and np.linalg.norm(Ps_new) < 50:
                Ps.append(Ps_new[:3])
            else:
                bad_idx.append(lm)
        
        Ps = np.array(Ps)
        return T, Ps, bad_idx
            
    def _register_image(self, cam_idx, im_landmarks):
        print(f"Connecting Camera {cam_idx}")
        
        # Sort through seen landmarks
        landmarks_all = np.vstack((self.landmarks, self.landmarks_new))

        # BFMatcher with default params
        knn_matches = self.match.match(landmarks_all[:,4:], im_landmarks[:,4:])
        knn_matches = sorted(knn_matches, key = lambda x : x.distance)

        # Apply ratio test
        match_idx = np.array([[m.queryIdx, m.trainIdx] for m in knn_matches])
        kp1_match = landmarks_all[match_idx[:,0], 2:4]
        kp2_match = im_landmarks[match_idx[:,1], 2:4]
        print("\t", match_idx.shape[0], "Starting # matches")

        # Ransac & find essential matrix
        # TODO: This is bad b/c it's being done with multiple images at a time
        # TODO: Choose one with most matches? (probably previous in most cases)
        # TODO: Or do it for each one?
        E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, self.K[:3,:3], method=cv2.RANSAC, threshold=1.5, prob=0.99)
        inlier = inlier.flatten().astype('bool')
        match_idx = match_idx[inlier]  
        print("\t", match_idx.shape[0], "After Essential matrix RANSAC")

        # If it's the first iteration
        if self.num_lm == 0:
            n = match_idx.shape[0]
            lm = np.arange(n)
            self.landmarks_seen = np.vstack((landmarks_all[match_idx[:,0]], im_landmarks[match_idx[:,1]]))
            self.landmarks_seen[:n,0] = lm
            self.landmarks_seen[n:,0] = lm
            a = np.lexsort((self.landmarks_seen[:,1], self.landmarks_seen[:,0]))
            self.landmarks_seen = self.landmarks_seen[a]
            return E, 0, match_idx.shape[0]
            
        else:
            # Get new landmarks
            old_lm = match_idx[:,0] < self.num_lm
            new_lm = match_idx[:,0] >= self.num_lm

            # Make array with old landmarks
            n_old = np.sum(old_lm)
            old_lm_numbers = landmarks_all[match_idx[old_lm,0],0]
            old_lm_matches = im_landmarks[match_idx[old_lm,1]]
            old_lm_matches[:,0] = old_lm_numbers
            print("\t", n_old, "Existing landmarks")

            # Make array with new landmarks
            n_new = np.sum(new_lm)
            lm = np.arange(self.num_lm, self.num_lm + n_new)
            new_lm_matches = np.vstack(((landmarks_all[match_idx[new_lm,0]], im_landmarks[match_idx[new_lm,1]])))
            new_lm_matches[:n_new,0] = lm
            new_lm_matches[n_new:,0] = lm
            print("\t", n_new, "New landmarks")

            # Put all togehter & sort
            self.landmarks_seen = np.vstack((self.landmarks_seen, old_lm_matches, new_lm_matches))
            a = np.lexsort((self.landmarks_seen[:,1], self.landmarks_seen[:,0]))
            self.landmarks_seen = self.landmarks_seen[a]

            # Delete landmarks from landmark new we just saw, and add in the ones from this image
            self.landmarks_new = np.delete(self.landmarks_new, match_idx[new_lm,1]-self.num_lm, axis=0)
            not_seen = np.delete(im_landmarks, match_idx[:,1], axis=0)
            self.landmarks_new = np.vstack((self.landmarks_new, not_seen))
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

    for i in images[:20]:
        im1 = read_image(i, scale_percent)
        sfm.add_image(im1)
        
    print(sfm.landmarks_seen.shape)