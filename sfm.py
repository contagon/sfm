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
np.set_printoptions(suppress=True, precision=3, edgeitems=30, linewidth=100000)

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
        self.seen = None
        self.kp = []
        self.des = []
        self.connections = connections
        
        self.feat = cv2.SIFT_create()
        # Enforce unique matching
        self.match = cv2.BFMatcher(crossCheck=True)
        
        self.Ts = []
        self.K = K
        self.Ps = np.zeros((0,3))
        
    # TODO: Figure out how to store measurements somewhere...

    def add_image(self, im):
        # Assumes images are added sequentially...

        # First, we find keypoints
        kp, des = self.feat.detectAndCompute(im,None)
        kp = np.array([k.pt for k in kp])
        
        self.kp.append(kp)
        self.des.append(des)
    
        this_idx = len(self.kp)-1
        if this_idx == 0:
            self.Ts.append(np.eye(4))
        else:
            # Next, we connect images together
            # TODO: Fix so you can check with multiple images at same time
            E, repeated_lm, new_lm = self._register_image(this_idx-1, this_idx)
            
            # Get an estimate for this frame & new landmarks
            if repeated_lm == 0:
                T, Ps = self._pose_from_E(E, new_lm, this_idx-1, this_idx)
            else:
                T, Ps, bad_idx = self._pose_from_pts(new_lm, this_idx)
                self.seen = np.delete(self.seen, bad_idx, 0)

            # Update everything
            self.Ts.append(T)
            self.Ps = np.append(self.Ps, Ps, 0)
            assert self.Ps.shape[0] == self.seen.shape[0]

            # Run a small optimization with this new information
        
    @property
    def _zs(self):
        # Organize measurements a  bit better
        meas = List()
        for i,col in enumerate(self.seen.T):
            lm_idx = np.nonzero(col)[0]
            kp_idx = col[lm_idx]
            kp = self.kp[i][kp_idx]
            meas.append((lm_idx, kp))
            
        return meas
    
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
        
        
    def _register_image(self, im_idx1, im_idx2):
        print(f"Connecting Image {im_idx1} -> {im_idx2}")
        
        # BFMatcher with default params
        knn_matches = self.match.match(self.des[im_idx1], self.des[im_idx2])
        knn_matches = sorted(knn_matches, key = lambda x : x.distance)

        # Apply ratio test
        match_idx = np.array([[m.queryIdx, m.trainIdx] for m in knn_matches])
        kp1_match = self.kp[im_idx1][match_idx[:,0]]
        kp2_match = self.kp[im_idx2][match_idx[:,1]]
        print("\t", match_idx.shape[0], "Starting # matches")

        # Ransac & find essential matrix
        E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, self.K[:3,:3], method=cv2.RANSAC, threshold=1.5, prob=0.99)
        inlier = inlier.flatten().astype('bool')
        match_idx = match_idx[inlier]  
        print("\t", match_idx.shape[0], "After Essential matrix RANSAC")

        # If it's the first iteration
        if self.seen is None:
            self.seen = match_idx.astype('int')
            return E, 0, match_idx.shape[0]
            
        else:
            # Get kp indices that were seen before
            im1_kp_seen = self.seen[:,im_idx1]
            im1_kp_seen = im1_kp_seen[im1_kp_seen > 0]

            # Add column to matrix if needed
            if self.seen.shape[1]-1 < im_idx2:
                self.seen = np.append(self.seen, np.zeros((self.seen.shape[0], 1), dtype='int'), 1)

            # Put in landmarks that were seen again
            seen_before = np.zeros(match_idx.shape[0], dtype='bool')
            for i in range(match_idx.shape[0]):
                lm_idx = np.where(match_idx[i,0] == im1_kp_seen)[0]
                if lm_idx.shape[0] > 0:
                    self.seen[lm_idx[0],im_idx2] = match_idx[i,1]
                    seen_before[i] = True
            print("\t", np.sum(seen_before), "Existing landmarks")

            # Put in new landmarks
            new_matches = match_idx[~seen_before]
            new_size = new_matches.shape[0]
            self.seen = np.append(self.seen, np.zeros((new_size, self.seen.shape[1]), dtype='int'), 0)
            self.seen[-new_size:,[im_idx1, im_idx2]] = new_matches
            print("\t", new_matches.shape[0], "New landmarks")
            
            return E, np.sum(seen_before), new_size

        print()
        
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

    for i in images[:8]:
        im1 = read_image(i, scale_percent)
        sfm.add_image(im1)