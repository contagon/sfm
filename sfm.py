import numpy as np
import cv2
from numba.typed import List
import open3d as o3d

from jacobian import jac
from manifold import SO3_from_vec
from optimize import levenberg_marquardt
from cv import residuals

class StructureFromMotion():
    def __init__(self, K, feat="sift"):
        # setup feature matching
        self.feat_type = feat
        if feat == "sift":
            self.feat = cv2.SIFT_create(nfeatures=10000)
            # Use both matchers to enforce unique matching
            self.flann_matcher = cv2.FlannBasedMatcher()
            self.bf_matcher = cv2.BFMatcher(crossCheck=True)
        elif feat == "orb":
            self.feat = cv2.ORB_create(nfeatures=10000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # landmark index, camera index, kp, pixel values, des
        feat_size = 128 if self.feat_type == "sift" else 32
        self.measurements = np.zeros((0,1+1+2+3+feat_size), dtype='float32')
        self.landmarks_new = np.zeros((0,1+1+2+3+feat_size), dtype='float32')
        self.num_cam = 0
        self.im = []
        

        # These transform world coordinates to camera coordinates. 
        # To plot, we plot the inverse of this
        self.Ts = np.zeros((0,4,4))
        self.K = K
        self.Ps = np.zeros((0,3))

        # For visualization
        self.vis = None
        self.pc = None
        self.frames = []

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
        self.im.append(im)
        # First, we find keypoints
        kp, des = self.feat.detectAndCompute(im,None)
        kp = np.array([k.pt for k in kp])

        # extract pixels values
        kp_int = np.round(kp).astype('int')
        pixels = im[kp_int[:,1], kp_int[:,0]]
        if pixels.dtype == 'uint8':
            pixels = pixels / np.max(pixels)
        
        landmarks_im = np.hstack([
                            np.full((kp.shape[0], 1), np.NaN),
                            np.full((kp.shape[0], 1), self.num_cam), 
                            kp,
                            pixels,
                            des
                        ]).astype('float32')
    
        if self.num_cam == 0:
            self.Ts = np.eye(4)[None,:,:]
            self.landmarks_new = landmarks_im
        else:
            # Next, we connect images together
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

    def plot(self, block=True):
        # Flips everything so it's easily viewable by default
        flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

        # If we're just starting
        if self.vis is None:
            # Make point cloud from our points
            self.pc = o3d.geometry.PointCloud()

            # Make visualization window
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("SfM")
            self.vis.add_geometry(self.pc)
            
        # Update point cloud
        self.pc.points = o3d.utility.Vector3dVector(self.Ps)
        self.pc.colors = o3d.utility.Vector3dVector(self.pixels)
        self.pc.transform(flip)
        self.vis.update_geometry(self.pc)

        # Update frames
        for i, T in enumerate(self.Ts):
            # Make new coordinate frame
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh.transform(np.linalg.inv(T))
            mesh.transform(flip)

            # Remove old one if it's there
            if i < len(self.frames):
                self.vis.remove_geometry(self.frames[i])
                self.frames[i] = mesh
            else:
                self.frames.append(mesh)

            # Add to visualization
            self.vis.add_geometry(mesh)

        # Update visualizer
        self.vis.poll_events()
        self.vis.update_renderer()
    
        if block:
            self.vis.run()
        
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

        # Organize measurements a bit better
        results = List((lm_sorted[idx[i]:idx[i+1]].astype('int'), mm_sorted[idx[i]:idx[i+1]]) for i in range(len(idx)-1))
        return results

    @property
    def pixels(self):
        # Get where to split everything
        lm = self.measurements[:,0]
        idx = np.where(np.roll(lm,1)!=lm)[0]
        idx = np.append(idx, lm.shape[0])

        # start averaging pixels :)
        pixels = np.array([np.mean(self.measurements[idx[i]:idx[i+1],4:7], axis=0)  for i in range(len(idx)-1)])
        # flip from bgr to rgb
        pixels[:,[0,2]] = pixels[:,[2,0]]
        return pixels

    def optimize(self, tol=1e-4, max_iters=50, verbose=10, line_start=""):
        self.K, self.Ts, self.Ps = levenberg_marquardt(
                                                residuals, 
                                                self.K, 
                                                self.Ts, 
                                                self.Ps, 
                                                self._zs, 
                                                jac, 
                                                max_iters=max_iters, 
                                                lam=0.01, 
                                                lam_multiplier=4, 
                                                tol=tol, 
                                                verbose=verbose,
                                                line_start=line_start
                                            )

    def save(self, filename):
        np.savez(filename, K=self.K, P=self.Ps, T=self.Ts, pixels=self.pixels)

    def _add_measurements(self, new_mm):
        # Put all togehter & sort
        self.measurements = np.vstack((self.measurements, new_mm))
        # Sort by landmark, then by camera
        a = np.lexsort((self.measurements[:,1], self.measurements[:,0]))
        self.measurements = self.measurements[a]
    
    def _pose_from_E(self, E, mm1, mm2):
        # Extract information we need for measurements
        im1_idx = int(mm1[0,1])
        kp1 = mm1[:,2:4]
        kp2 = mm2[:,2:4]

        # Make sure we're only estimating from one other camera
        assert len(np.unique(mm1[:,1])) == 1

        # recover pose (this pose returns x^2 = T x^1)
        points, R, t, inliers = cv2.recoverPose(E, kp1, kp2, self.K[:3,:3])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t.flatten()

        # needs to be T^i_w not T^i_{i-1} (multiply by T^{i-1}_w)
        T = T@self.Ts[im1_idx]

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
        # Get kps and landmark indices for previously seen landmarks
        kps_old = old_lm[:,2:4]
        lms_old = old_lm[:,0].astype('int')
        
        # Use the estimating point estimates to find relative pose
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(self.Ps[lms_old], kps_old, self.K[:3,:3], None, reprojectionError=8, confidence=.99)
        inliers = inliers.flatten()
        T = np.eye(4)
        T[:3,:3] = SO3_from_vec(rvec.flatten())
        T[:3,3] = tvec.flatten()
        print("\t", inliers.shape[0], " existing landmarks are kept when running PnP")

        # Put everything where it needs to go
        self._add_measurements(old_lm[inliers])
        bad_mm = np.delete(old_lm, inliers, axis=0)
        bad_mm[:,0] = np.NaN
        self.landmarks_new = np.vstack((self.landmarks_new, bad_mm))


        # Triangulate all the other ones
        all_Ps = []
        outliers = []
        cams_with_matches = np.sort(np.unique(new_lm1[:,1])).astype('int')
        for cam in cams_with_matches:
            # Get this cameras indices
            this_cam_matches = np.where(new_lm1[:,1] == cam)[0]
            cam_lm1 = new_lm1[this_cam_matches]
            cam_lm2 = new_lm2[this_cam_matches]

            # Triangulate
            Ps = cv2.triangulatePoints(self.K@self.Ts[cam], self.K@T, cam_lm1[:,2:4].T, cam_lm2[:,2:4].T)
            Ps /= Ps[3] # comes out with each point as a different column
            # Make sure it's not behind either camera (have to switch frames) or crazy far away
            inliers = np.vstack(( (self.Ts[cam]@Ps)[2] > 0, (T@Ps)[2] > 0, np.linalg.norm(Ps[:3], axis=0) < 15 )).all(axis=0)
            all_Ps.append(Ps.T[inliers,:3])

            # Save inlier measurements
            n = np.sum(inliers)
            lm = np.arange(self.num_lm, self.num_lm+n)
            mm_good = np.vstack((cam_lm1[inliers], cam_lm2[inliers]))
            mm_good[:n,0] = lm
            mm_good[n:,0] = lm
            self._add_measurements(mm_good) 

            # Save outlier indices
            outliers.append( this_cam_matches[~inliers] )

        # Remove outliers, insert into unmatched array
        outliers = np.concatenate(outliers)
        bad_mm = np.vstack((new_lm1[outliers], new_lm2[outliers]))
        self.landmarks_new = np.vstack((self.landmarks_new, bad_mm))

        all_Ps = np.vstack(all_Ps)
        print("\t", all_Ps.shape[0], "new landmarks were kept when triangulating")

        return T, all_Ps
    
    def _match(self, des1, des2):
        if self.feat_type == "sift":
            # Match with FLANN, then crosscheck with BF
            match_flann = self.flann_matcher.match(des1, des2)
            match_flann = np.array([[m.queryIdx, m.trainIdx] for m in match_flann])
            match_flann1 = match_flann[:,0]
            match_flann2 = np.unique(match_flann[:,1]) # since there's no crosscheck yet, these may not be unique
            des1 = des1[match_flann1]
            des2 = des2[match_flann2]
            
            match_bf = self.bf_matcher.match(des1, des2)
            match_bf = sorted(match_bf, key = lambda x : x.distance)
            match_bf = np.array([[m.queryIdx, m.trainIdx] for m in match_bf])
            match_idx = np.column_stack((match_flann1[match_bf[:,0]], match_flann2[match_bf[:,1]]))

        elif self.feat_type == "orb":
            match_bf = self.bf_matcher.match(des1.astype('uint8'), des2.astype('uint8'))
            match_bf = sorted(match_bf, key = lambda x : x.distance)
            match_idx = np.array([[m.queryIdx, m.trainIdx] for m in match_bf])

        return match_idx

    def _register_image(self, cam_idx, landmarks_im):
        print(f"Connecting Camera {cam_idx}")
        print(f"\t Found {landmarks_im.shape[0]} keypoints")

        # use only unmatched kp from previous camera
        new = self.landmarks_new[ self.landmarks_new[:,1] == cam_idx-1 ] 
        # Only use matched landmarks from previous 4 camers
        previous = self.landmarks
        previous_mask = previous[:,1] == cam_idx-1
        for i in range(2,8):
            previous_mask = np.logical_or(previous_mask, previous[:,1] == cam_idx-i)
        previous = previous[previous_mask]

        landmarks_all = np.vstack((new, previous))

        # Match landmarks
        des1 = landmarks_all[:,7:].copy()
        des2 = landmarks_im[:,7:].copy()
        match_idx = self._match(des1, des2)

        # Get matches
        landmarks1_matched = landmarks_all[match_idx[:,0]]
        landmarks2_matched = landmarks_im[match_idx[:,1]]
        print("\t", match_idx.shape[0], "Starting # matches")

        # match indices to remove later
        outliers = []

        # Iterate through all camera matches, starting with first one 
        cams_with_matches = np.sort(np.unique(landmarks1_matched[:,1])).astype('int')
        for cam in cams_with_matches:
            # Get keypoints
            this_cam_matches = np.where(landmarks1_matched[:,1] == cam)[0]
            if len(this_cam_matches) > 6:
                kp1_match = landmarks1_matched[this_cam_matches, 2:4]
                kp2_match = landmarks2_matched[this_cam_matches, 2:4]

                # Ransac & find essential matrix
                E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, self.K[:3,:3], method=cv2.RANSAC, threshold=1, prob=0.999)
                inlier = inlier.flatten().astype('bool')
                outliers.append( this_cam_matches[~inlier] )
                print(f"\t Cam {cam} had {inlier.shape[0]} matches, {np.sum(inlier)} after Essential matrix RANSAC")
            else:
                outliers.append(this_cam_matches)
                # print(f"\t Cam {cam} had {this_cam_matches.shape} matches")

        # remove all outliers we found
        match_idx = np.delete(match_idx, np.concatenate(outliers), axis=0)

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

            # Delete landmarks from landmark new we just saw, and add in the ones that didn't match in image
            new_lm_matches_idx = np.where(self.landmarks_new[:,1] == cam_idx-1)[0][ match_idx[new_lm,0] ]
            self.landmarks_new = np.delete(self.landmarks_new, new_lm_matches_idx, axis=0) # remove matches
            not_seen_im = np.delete(landmarks_im, match_idx[:,1], axis=0) # remove all matches from last image
            self.landmarks_new = np.vstack((self.landmarks_new, not_seen_im))

            return E, old_lm_matches, new_lm_matches
        
    def close_loop(self):
        cam1_idx = self.num_cam-1
        cam2_idx = 0
        print(f"Connecting Cam {cam1_idx} and Cam {cam2_idx} for a loop closure..")

        # Get all matches in both
        landmarks1 = np.vstack((
                            self.landmarks_new[self.landmarks_new[:,1] == cam1_idx],
                            self.measurements[self.measurements[:,1] == cam1_idx],
                        ))
        landmarks2 = np.vstack((
                            self.landmarks_new[self.landmarks_new[:,1] == cam2_idx],
                            self.measurements[self.measurements[:,1] == cam2_idx],
                        ))
        
        # Match them
        des1 = landmarks1[:,7:].copy()
        des2 = landmarks2[:,7:].copy()
        match_idx = self._match(des1, des2)

        # Find inliers
        kp1_match = landmarks1[match_idx[:,0], 2:4]
        kp2_match = landmarks2[match_idx[:,1], 2:4]
        E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, self.K[:3,:3], method=cv2.RANSAC, threshold=1, prob=0.999)
        inlier = inlier.flatten().astype('bool')
        
        match_idx = match_idx[inlier]
        print(f"\t {inlier.shape[0]} matches, {np.sum(inlier)} after Essential matrix RANSAC")

        # Seperate into which have been seen or not before
        matched1 = landmarks1[match_idx[:,0]]
        matched2 = landmarks2[match_idx[:,1]]
        new_lm = np.logical_and(np.isnan(matched1[:,0]), np.isnan(matched2[:,0]))
        old_lm = np.logical_xor(np.isnan(matched1[:,0]), np.isnan(matched2[:,0]))
        old_old_lm = np.vstack((~np.isnan(matched1[:,0]), ~np.isnan(matched2[:,0]), matched1[:,0]==matched2[:,0])).all(axis=0) # make sure not to include matches we already have!


        # Renumber landmarks that were seen in both images! (default to smaller landmark number) (will be sorted later)
        for m, n in zip(matched1[old_old_lm,0], matched2[old_old_lm,0]):
            self.measurements[self.measurements[:,0] == m, 0] = n
        self.Ps = np.delete(self.Ps, matched1[old_old_lm,0].astype('int'), axis=0)
        print("\t", np.sum(old_old_lm), "Landmarks seen in both images before")

        # Make array with landmarks that were seen once before
        old_lm_matches1 = matched1[old_lm]
        old_lm_matches2 = matched2[old_lm]
        mask1 = np.isnan(old_lm_matches1[:,0])
        old_lm_matches1[mask1,0] = old_lm_matches2[mask1,0]
        mask2 = np.isnan(old_lm_matches2[:,0])
        old_lm_matches2[mask2,0] = old_lm_matches1[mask2,0]

        old_lm = np.vstack((old_lm_matches1, old_lm_matches2))
        self._add_measurements(old_lm)
        print("\t", old_lm_matches1.shape[0], "Landmarks seen in one image before")


        # TODO: self.landmarks_new was NOT updated from all this!

        # TODO: This only connects existing landmarks, add to it to get all landmarks that hadn't been seen in either image before
        # In my experience there was very few of them
        # Make array with new landmarks
        # new_lm_matches = (landmarks_all[match_idx[new_lm,0]], landmarks_im[match_idx[new_lm,1]])
        # print("\t", np.sum(new_lm), "New landmarks")

        # # Delete landmarks from landmark new we just saw, and add in the ones that didn't match in image
        # not_seen_all = np.delete(landmarks_all, match_idx[:,0], axis=0) # remove matches
        # not_seen_all = not_seen_all[ np.isnan(not_seen_all[:,0]) ] # keep only kp that don't have landmarks
        # not_seen_im = np.delete(landmarks_im, match_idx[:,1], axis=0) # remove all matches from last image
        # self.landmarks_new = np.vstack((not_seen_all, not_seen_im))

