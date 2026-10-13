# ViPE environment (camera-trajectory diversity)

The Cam / CamEnd columns of the video table use camera poses estimated by
[ViPE](https://github.com/nv-tlabs/vipe) (`nvidia-vipe` 1.2.0) with a pose-only pipeline. ViPE
compiles a CUDA extension at install time, so it lives in its own Python 3.10 environment, built
against torch 2.7.1 (CUDA 12.6) and a CUDA 12.6 toolkit. Builds against a CUDA 13 toolkit do not
compile.

## Build

```bash
# 1. CUDA 12.6 toolkit (nvcc) for the extension build; the runfile installs it into a user directory.
#    https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
sh cuda_12.6.3_560.35.05_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-12.6 \
    --defaultroot=$HOME/cuda-12.6 --no-man-page --no-drm --override
export CUDA_HOME=$HOME/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=9.0        # compute capability of your GPU (9.0 = H100)

# 2. Python 3.10 environment with torch 2.7.1 + CUDA 12.6 and the build tools
python3.10 -m venv venvs/vipe
venvs/vipe/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
venvs/vipe/bin/pip install setuptools==78.1.0 wheel==0.48.0 ninja==1.13.2 numpy==2.2.6

# 3. ViPE from its PyPI source distribution, compiled against the installed torch
venvs/vipe/bin/pip install --no-build-isolation nvidia-vipe==1.2.0 -c envs/constraints-vipe.txt
```

`envs/constraints-vipe.txt` pins every other package to the version of the reference environment.

## Pose-only pipeline

ViPE's default pipeline also runs instance segmentation and dense depth alignment, which camera
trajectories do not need: the poses and intrinsics come from its SLAM stage. Add this pipeline
config to the installed package:

```bash
CFG=$(venvs/vipe/bin/python -c "import os, vipe; print(os.path.join(os.path.dirname(vipe.__file__), '_configs', 'pipeline'))")
cat > "$CFG/pose_only.yaml" <<'EOF'
defaults:
  - default

# pose-only: SLAM camera poses + intrinsics, no instance segmentation, no dense depth alignment
init:
  instance: null

post:
  depth_align_model: null
EOF
sha256sum "$CFG/pose_only.yaml"   # 5f35ed4395dc34767cbb1b8ef95ace1fa358611b428179b0a1412e9f32a25394
```

The repository ships the same file as `evaluation/video/vipe_pose_only.yaml` (same sha256), so it
can be copied instead:

```bash
cp evaluation/video/vipe_pose_only.yaml "$CFG/pose_only.yaml"
```

## Running

The CUDA 12.6 toolkit must stay on `PATH` / `LD_LIBRARY_PATH` at run time. ViPE downloads its own
model weights on first use.

```bash
export PATH=$HOME/cuda-12.6/bin:$PATH LD_LIBRARY_PATH=$HOME/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
venvs/vipe/bin/vipe infer <clip.mp4> -p pose_only -o <output dir>
```

The evaluation scripts call `vipe infer` this way for every clip of the 30-prompt subset
`prompts/vbench_subset30.txt`. `evaluation/video/camtraj_vipe.py` runs ViPE as `$VIPE_BIN` (default:
the `vipe` script next to its interpreter, else `vipe` on `PATH`). When `VIPE_CUDA_HOME` is set, for
example to `$HOME/cuda-12.6`, it puts that toolkit's `bin/` and `lib64/` in front of `PATH` /
`LD_LIBRARY_PATH` of the ViPE process, so the exports above are not needed.
