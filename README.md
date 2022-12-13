![](results/point_cloud_far.png)

# Structure from Motion

This project was done for Robomath (16811) @ CMU in Fall 2022. All the bookkeeping, initialization, and optimization (including analytical Jacobians!) were done by hand in python. Usage as follows:

- `run.py` runs SfM. Run `python run.py --help` for input/output options. For example, after downloading the moose data (see line below), it can be ran quickly with the first 15 images at 50% scale and live plotting with `python run.py -p -s 25 -n 15`. Note live plotting is not interactive until SfM has finished, and output checkpoints will be saved to `results/` by default. Also, currently the initial estimates for the intrinsics are set inside the file.
- `download_data.py` downloads and extracts the data for the moose ornament example into the `data/` folder. Make sure you install dependencies for this first (see below).
- `plot.py` shows the 3D point cloud. Run `python plot.py results/point_cloud.npz` to see the resulting moose ornament point cloud.
- `slide_images.py` is full of some helpers to create images for the slides/writeups.
- `point_cloud_to_mesh.py` was a first attempt at some surface reconstruction from the dense point cloud. Requires more tuning.

The majority of the SfM code can be found in the [sfm](sfm) directory.

**More technical info can be found in the [writeup](writeup.pdf) and [slides](slides.pdf).**


## Installation

I recommend installation into a conda environment. A simple command to make and activate a new environment is
```
conda create -n sfm python=3.10
conda activate sfm
```
Then to install all required dependencies, run
```
pip install -r requirements.txt
```