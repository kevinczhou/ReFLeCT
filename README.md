# High-speed 4D fluorescence light field tomography of whole freely moving organisms
<center><img src="/media/ReFLeCT_overview.png" alt="ReFLeCT" width="800"/></center>
We present <ins>R</ins>eflective <ins>F</ins>ourier <ins>L</ins>ight field <ins>C</ins>omputed <ins>T</ins>omography (ReFLeCT), a new high-throughput 4D imaging technique based on an array of cameras and a large parabolic mirror objective. The cameras can capture synchronized video at up to 120 fps from multiple views, spanning nearly 2π steradians, from which we can computationally reconstruct 4D volumetric videos of fluorescently-labeled, freely moving organisms, such as fruit fly and zebrafish larvae (even multiple in the same field of view). This repository contains the code for performing the 4D reconstructions. Other related repositories:

- Accompanying repository dedicated to interactive 4D visualization of the reconstructions using napari (https://github.com/kevinczhou/ReFLeCT-4D-visualization). 
- The ReFLeCT forward model is based on our earlier work with 3D OCRT (https://github.com/kevinczhou/3d-ocrt), which also used a parabolic mirror to obtain very wide view angles.


For more details, see our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.09.16.609432v1).

## Data
We provide the raw multi-view video data for four of the samples in our paper (Figs. 3-6 in the bioRxiv preprint, Figs. 4-7 in the *Optica* paper), which can be downloaded [here](https://doi.org/10.7924/r4zs30t4h). Associated with each sample is
- `raw_video.nc`: raw video file (10-sec recordings, 50 GB each)
- `calibration_parameters.mat`: calibration file, providing the parameters specifying the ray trajectories
- `occupancy_grid.npz`: a boolean tensor, crudely specifying the organism vs. background


## Setting up your compute environment
Follow the Docker setup instructions [here](https://github.com/kevinczhou/3D-RAPID?tab=readme-ov-file#setting-up-your-environment), using the Docker file provided here (`media/dockerfile`).

## Usage
Download the data from [here](https://doi.org/10.7924/r4zs30t4h), preserving the file path structure (adjust `data_path` as needed). Next, run `ReFLeCT_reconstruction.ipynb`. In the second cell, select the sample via `sample_id` and set whether you want to reconstruct a single video frame or the full video (sequentially) via `optimize_single_frame`. Depending on the latter choice, run the code in either the "Single-frame optimization" or "Frame by frame video reconstructions" section. Note that the code requires a GPU, as some functions have different behavior on GPU vs CPU. Adjust `batch_size` so that the model fits on your GPU (we used a 48-GB GPU). 

When reconstructing the full video, select which frames of the video you want to reconstruct by adjusting `video_frames`. Right now, the notebook loads all the raw video frames specified by `video_frames`, even though the reconstructions are done one time point at a time. The code could be modified to sequentially load the frames as they are needed, but for now, if your CPU RAM is limited, reduce the number of frames in `video_frames`.
