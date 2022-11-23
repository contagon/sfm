from sfm import StructureFromMotion
import numpy as np
import argparse
import os
from cv import read_image

np.set_printoptions(suppress=True, precision=3) #, edgeitems=30, linewidth=100000)

def run(scale, plot, in_files, outfile, num_images):
    # Initialize K
    K_init = np.array([[3271.7198,    0.,     1539.6885, 0],
             [   0.,     3279.7956, 2027.496, 0],
             [   0.,        0.,        1.,0    ]])
    K_init[:2] *= scale/100

    # Get all file locations
    valid_imgs = [".jpg", ".png"]
    images = [os.path.join(in_files, f) for f in sorted(os.listdir(in_files)) if os.path.splitext(f)[1].lower() in valid_imgs]

    # The big dog
    sfm = StructureFromMotion(K_init)

    # Iterate through, lightly optimizing as we add each image
    for i, file in enumerate(images[:num_images]):
        # Add image to sfm
        im1 = read_image(file, scale)
        sfm.add_image(im1)

        # Optimize & plot
        if sfm.num_cam > 1:
            print(f"\t Optimizing cam {i} results...")
            sfm.optimize(tol=1, max_iters=10, line_start="\t\t")
            if plot:
                sfm.plot(block=False)

            print()

    # More accurate optimization
    print("Optimizing one last time...")
    sfm.optimize(tol=1e-3, max_iters=100, line_start="\t")

    # Cleanup
    if outfile:
        sfm.save(outfile)
    if plot:
        sfm.plot()

    print()
    print("~~~~~~~ Finished! ~~~~~~~~~~")


if __name__ == "__main__":
    # Parse through arguments
    # TODO: Way to read initial K in?
    parser = argparse.ArgumentParser(description="Structure from Motion")
    parser.add_argument("-i", "--in_files", type=str, default="data/statue", help="Folder that images are stored in. Images are assumed to be sequentially named.")
    parser.add_argument("-s", "--scale", type=int, default="25", help="Percentage to scale images down to.")
    parser.add_argument("-n", "--num_images", type=int, default=None, help="Use first n images. Defaults to all images.")
    parser.add_argument('-p', '--plot', action='store_true', help='Plot data when each image is read in, as well as at end.')
    parser.add_argument("-o", "--outfile", type=str, default="out.npz", help="Npz file to save resulting intrinsics, poses, 3d points, and 3d pixel colors to. Defaults to out.npz.")
    args = vars(parser.parse_args())
    run(**args)