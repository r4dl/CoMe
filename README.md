<h1 align="center">Confidence-Based Mesh Extraction from 3D Gaussians</h1>

<p align="center">
  <a href="https://r4dl.github.io/CoMe/">
    <img src="https://img.shields.io/badge/Project-Page-darkblue" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2603.24725">
    <img src="https://img.shields.io/badge/arXiv-2603.24725-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://cloud.tugraz.at/index.php/s/tRz85cJsRQGJX4q">
    <img src="https://img.shields.io/badge/Data-Pointclouds-darkgreen" alt="Point Clouds">
  </a>
  <a href="https://cloud.tugraz.at/index.php/s/ysRAxopxHqfHxzn">
    <img src="https://img.shields.io/badge/Data-Meshes-darkorange" alt="Meshes">
  </a>
  <a href="https://youtu.be/dL-77OGCCjw">
    <img src="https://img.shields.io/badge/Video-YouTube-red" alt="Video">
  </a>
</p>

<h3 align="center">arXiv 2026</h3>

<h4 align="center">
    <a href="https://r4dl.github.io/">Lukas Radl*</a><sup>1</sup> ·
    <a href="https://felixwindisch.github.io/">Felix Windisch*</a><sup>1</sup> ·
    <a href="https://scholar.google.com/citations?user=3yD4NZgAAAAJ&hl=en&oi=ao">Andreas Kurz*</a><sup>1</sup><br>
    <a href="https://derthomy.github.io/">Thomas Köhler</a><sup>1</sup> ·
    <a href="https://steimich96.github.io/">Michael Steiner</a><sup>1</sup> ·
    <a href="https://www.markussteinberger.net/">Markus Steinberger</a><sup>1,2</sup>
</h4>

  <div align="center">
    <p>
      <sup>1</sup> Graz University of Technology 🇦🇹<br>
      <sup>2</sup> Huawei Technologies 🇦🇹
    </p>
  </div>

## Overview

**CoMe** is a method for **unbounded mesh extraction**, using 3D Gaussians. Compared to recent methods, CoMe faithfully balanced photometric and geometric losses via a confidence-based framework, enabled fast, detailed mesh extraction.
For a more visual overview, *cf.* our [project page](https://r4dl.github.io/CoMe/).

<div align="center">
  <img src="assets/output.gif"/>
</div>

## News

> - **April 23, 2026** — Code/Assets release. We updated the paper on [arXiv](https://arxiv.org/abs/2603.24725), and fixed a minor bug regarding the normal variance loss; results are not affected.

## Code

> Find all instructions for **running our code** here!

<details>
<summary><strong>Setup</strong></summary>

```bash
# Clone the repository
git clone https://github.com/r4dl/CoMe.git
cd CoMe

# Create a conda environment
# default settings: torch > 2.1, cuda 12.1 (tested)
conda env create --file environment.yml
conda activate come

# Install the remaining dependencies
pip install submodules/simple-knn/ --no-build-isolation
pip install submodules/diff-gaussian-rasterization/ --no-build-isolation
# NEW: Custom Fused SSIM Implementation
pip install submodules/decoupled-fused-ssim/ --no-build-isolation
# Fused Implementation from Rahul for backwards compatibility
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
``` 

To extract meshes, install Tetra-Triangulation, based on [Tetra-NeRF](https://github.com/jkulhanek/tetra-nerf):
```bash
cd submodules/tetra-triangulation

cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5
# to build, it might be necessary for building to define the CUDA PATH
# export CPATH=/usr/local/<CUDA_VERSION>/targets/x86_64-linux/include:$CPATH
make
# Note: editable mode is required here
pip install -e . --no-build-isolation
``` 
> We have tested this implementation with **Ubuntu 22.04** and **CUDA 12.1**.

</details> 

<details>
<summary><strong>Data</strong></summary>

For our evaluation, we used the following datasets:

| Dataset Name | Link | Note |
|--------------|------|------|
| Tanks & Temples | [Download](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/blob/main/TNT_GOF.zip) | ⚠️ **See Instructions below!** |
| DTU | [Download](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) | ⚠️ **See Instructions below!** |
| Mip-NeRF 360 | [Download](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) | - |
| ScanNet++-v2 | [Download](https://scannetpp.mlsg.cit.tum.de/scannetpp/) | ⚠️ **See Instructions below!** |

The links redirects you to a download page!
We assume all data within the `data/` directory for our <strong>scripts</strong> to work. 
If your data lies somewhere else, modify `DATA_DIR` in [scripts/constants.py#L11](scripts/constants.py#L11).
```
data
├── TNT_GOF
│   ├── Barn
│   └── ...
├── DTU
├── SCN
└── m360
```

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>Tanks & Temples post-install</strong></summary>

> **Note**: For Tanks and Temples, additional care needs to be taken!

First, you need to rename `<SCENE>_COLMAP_SfM.log` to `<SCENE>_traj_path.log` for every scene!
Afterwards, visit the [download page for TNT](https://www.tanksandtemples.org/download/). For each scene, download everything and paste into the corresponding scene folder!

> **Now** your setup is good to go!

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>ScanNet++ post-install</strong></summary>

We only tested a small subset of all scenes, see [scripts/constants.py#L26](scripts/constants.py#L26).
To run our scripts, move these scenes directly into `<DATA_DIR>/SCN`.

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>DTU post-install</strong></summary>

Download both the `SampleSet` and the `Points` from [here](https://roboimagedata.compute.dtu.dk/?page_id=36). See `DTU_GT_DATA` in `scripts/run_dtu.py`.

</details>

</blockquote>

</details> 

</details> 

<details>
<summary><strong>Scripts</strong></summary>

We provide scripts to train, mesh/render and evaluate our method, using the same hyperparameters as reported in the paper/used in the evaluation.

> **Note**: There may be some noise in the final results; for <strong>convenience</strong>, we provide the [point clouds](https://cloud.tugraz.at/index.php/s/tRz85cJsRQGJX4q)/[meshes](https://cloud.tugraz.at/index.php/s/ysRAxopxHqfHxzn) we used for evaluation in our paper (Tanks and Temples only)!

```bash
# Training, Meshing (Marching Tets) and Evaluation for Tanks & Temples
python scripts/run_tnt.py
# Training, Meshing (Marching Tets) and Evaluation for ScanNet++ 
python scripts/run_scn.py    
# Training, Meshing (TSDF) and Evaluation for DTU 
python scripts/run_dtu.py    
# Training, Rendering and Evaluation for NVS (Mip-NeRF 360 by default)
python scripts/run_nvs.py 
``` 

> **Note**: To show the results, simple use the corresponding `show_*` script, *e.g.*, `python scripts/show_nvs.py`.

</details> 

<details>
<summary><strong>Training</strong></summary>

To train our method, use the `train.py` script, as in, *e.g.* [StopThePop](https://github.com/r4dl/StopThePop). To document **rasterizer settings**, we use `.json` files, located in the `configs/` directory.

```bash
# SOF default settings
python train.py --splatting_config configs/hierarchical.json -s <path to dataset>
```

> See [StopThePop](https://github.com/r4dl/StopThePop) or [SOF](https://github.com/r4dl/SOF) for more details!

The most important new hyperparameters live under `MeshingParams` in [arguments/__init__.py](arguments/__init__.py); non-default values for experiments are in the paper or in `scripts/run_*.py`.

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>SSIM-decoupled appearance</strong></summary>

- `use_vastgaussian_appearance` (default: `false`)
- `use_ssimdecoupled_appearance` (default: `false`, **Ours**)

> **Note**: Defaults to no appearance embedding (e.g. for Mip-NeRF): Use `--use_ssimdecoupled_appearance` for meshing!

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>Color confidence</strong></summary>

- `color_confidence` (default: `false`)
- `color_confidence_max` (default: `0.075`)
- `color_confidence_from_iter` (default: `500`)

> **Note**: Optional confidence weighting for color; all three flags live in [`MeshingParams`](arguments/__init__.py). Use `--color_confidence` for meshing!

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>Variance losses</strong></summary>

- `lambda_variance` (default: `0.0`)
- `variance_from_iter` (default: `15000`)
- `lambda_normal_variance` (default: `0.0`)
- `normal_variance_from_iter` (default: `15000`)

> **Note**: Losses are off by default; increase `lambda_*` after `*_from_iter` to use the auxiliary color / normal variance terms. 
Use `--lambda_variance 0.5 --lambda_normal_variance 0.005` for meshing!

</details>

</blockquote>

</details> 

<details>
<summary><strong>Meshing</strong></summary>

#### Synthetic (such as DTU)
For synthetic, single-object scenes (such as DTU), we use **TSDF fusion**, which can be run using
```bash
python extract_mesh_tsdf.py -m <MODEL_PATH>
```
> **Note**: By default, we use a `voxel_size` of `0.002`, but it can be modified via `--voxel_size`.

As a result, you will get the ply-file in `<MODEL_PATH>/test/ours_30000/tsdf.ply`.


#### Real-World (such as Tanks & Temples/ScanNet++)
Here, we use **Fast Marching Tetrahedra** (as proposed by [SOF](https://github.com/r4dl/SOF)), which are run using
```bash
python extract_mesh_tets.py -m <MODEL_PATH>
```

> **Hint**: If you run out-of-memory or obtain overly large meshes, consider adding `--opacity_cutoff_tetra <VAL>`, with `<VAL>` larger than `0.0039` (the default value). This will remove redundant, almost transparent primitives from the initial point set.

As a result, you will get the ply-file in `<MODEL_PATH>/test/ours_30000/mesh_faster_binary_search_7.ply`.

> **Note**: We marginally accelerated the mesh extraction process by no longer obtaining exact opacity values for the first iteration, compared to SOF.

> **Note**: You can (and probably should) inspect these meshes using our mesh viewer (`python mesh_viewer.py <PATH TO PLY FILE>`). See the [Visualization & Debugging](#visualization--debugging) section below for more details.

</details> 

<details>
<summary><strong>Evaluation</strong></summary>

All evaluation scripts for meshing are contained in `mesh_utils/`, whereas the evaluation scripts for novel view synthesis are in the base directory.

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>Tanks & Temples</strong></summary>

To evaluate your meshes for the Tanks & Temples dataset, use
```bash
python mesh_utils/eval_TNT.py \
--dataset-dir <DATASET> \
--ply-path <PATH TO MESH> \
--traj-path <TRAJ PATH LOG FILE> \
--out-dir <OUT DIR>
```
> **Note**: For the `<TRAJ PATH LOG FILE>`, we used the `<SCENE>_COLMAP_SfM.log` file you get from the [TNT_GOF download](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/blob/main/TNT_GOF.zip); see [Data](#data) for details.

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>ScanNet++</strong></summary>

To evaluate your meshes for the ScanNet++ dataset, use
```bash
python mesh_utils/eval_SCN.py \
--dataset-dir <DATASET> \
--ply-path <PATH TO MESH> \
--out-dir <OUT DIR>
```
> **Note**: By default, we use a $\tau$ of `0.05`, you can change this [here](mesh_utils/eval_SCN.py#L63).

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>DTU</strong></summary>

To evaluate your meshes for the DTU dataset, use
```bash
python mesh_utils/eval_DTU.py \
--instance_dir <PATH TO SCAN> \
--input_mesh <PATH TO MESH> \
--dataset_dir <PATH TO GT DATA> \
--vis_out_dir <OUT DIR>
```
> **Note**: The `GT DATA` needs to be downloaded separately from [this webpage](https://roboimagedata.compute.dtu.dk/?page_id=36); see [Data](#data) for details.

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>Novel View Synthesis (Mip-NeRF 360)</strong></summary>

To evaluate novel view synthesis, run
```bash
# render images
python render.py -m <MODEL DIRECTORY> --skip_train
# create metrics
python metrics.py -m <MODEL DIRECTORY>
```
This is the exact same workflow as in `scripts/run_nvs.py`.

Alternatively, you can also adapt the `run_nvs.py` script.

> **Note**: By default, we run Mip-NeRF 360 using the default settings; to modify this, modify the script:

```python
# modify these to test a different dataset
scenes = ...
factors = ...
TRAIN_DATA = ...
```

</details>

</blockquote>

</details> 

<details>
<summary><strong>Metrics</strong></summary>

These are the results for the latest run, using this codebase!

> **Note**: The numbers may vary slightly per-run, and this is not the original codebase we used; although a cleaned-up version!

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>Tanks and Temples</strong></summary>

> *Table: F1-Score evaluation*

| Metric    | Barn | Caterpillar | Courthouse | Ignatius | Meetingroom | Truck | Average |
|-----------|------|-------------|------------|----------|-------------|-------|---------|
| **Code (v1)** | 0.534 | 0.466 | 0.334 | 0.779 | 0.375 | 0.639 | **0.521** |
| **Paper** | 0.534 | 0.472 | 0.333 | 0.782 | 0.372 | 0.634 | **0.521** |

> **Hint**: Use `show_tnt.py` script to quickly get the metrics (after the corresponding `run`-script)!

</details>

</blockquote>

<blockquote style="border-left: solid 0.25em #d0d7de; margin: 0.5em 0 0.75em 0.75em; padding-left: 1em;">

<details>
<summary><strong>ScanNet++ (small)</strong></summary>

> *Table: F1-Score evaluation*

| Metric | 5a269b | 08bbbd | 39f36d | dc263d | ef18cf | fb564c | **Average** |
|--------|---------|--------|---------|---------|--------|-------|-------------|
| **Code (v1)** | 0.663 | 0.729 | 0.666 | 0.722 | 0.528 | 0.661 | **0.662** |
| **Paper** | 0.670 | 0.729 | 0.657 | 0.715 | 0.551 | 0.684 | **0.668** |

> **Note**: We remove the last 4 letters/digits of the scene name for a better layout.

</details>

</blockquote>

</details> 

<details>
<summary><strong>Visualization & Debugging</strong></summary>

Our visualization suite is built upon [Splatviz](https://github.com/Florian-Barthel/splatviz), and is fully self-contained within this repository.
To use it, first navigate to the `splatviz/` directory.

In it, run either
```bash
# if not already in splatviz
cd splatviz

# to attach to a currently running training session
python run_main.py --mode attach {--port <PORT>}

# to render a trained gaussian point cloud
# <PATH TO A POINT CLOUD FILE> must be a directory
python run_main.py --data_path <PATH TO A POINT CLOUD FILE>
```

With <strong>both</strong> (yes, both), open the `Render` tab to checkout different debug visualization modes (e.g. Depth/Normal/Transmittance/Confidence), modify rasterizer settings *on-the-fly* or just inspect the current scene.
<p align="center">
  <img src="assets/sofviz.png" alt="SOF Demo Teaser" width="95%" />
</p>

#### Inspecting Meshes
We additionally provide a **mesh viewer** to inspect triangulated meshes. To run, simply do
```bash
python mesh_viewer.py <PATH TO PLY FILE>
```
By default, normals are displayed. Checkout the CLI for more information! 

</details> 

## Licensing

This code has been built on top of <a href="https://github.com/r4dl/SOF">SOF</a>, which was built on top of <a href="https://github.com/r4dl/StopThePop">StopThePop</a>, and as such, is primarily licensed under the  <a href="LICENSE.md">"Gaussian Splatting License"</a>.
For more information, we refer to our <a href="NOTICE.md">Notice</a>.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{radl2026come,
author = {Radl, Lukas and Windisch, Felix and Kurz, Andreas and K{\"o}hler, Thomas and Steiner, Michael and Steinberger, Markus},
title = {{Confidence-Based Mesh Extraction from 3D Gaussians}},
year = {2026},
eprint = {2603.24725},
archivePrefix = {arXiv},
primaryClass = {cs.CV},
url = {https://arxiv.org/abs/2603.24725}, 
}</code></pre>
  </div>
</section>
