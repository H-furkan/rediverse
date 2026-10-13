"""ReDiverse: shared, backbone-agnostic helpers.

  rediverse.seeds       every seed rule (training noise, evaluation noise, direction latents,
                        ReDiverse-TF coefficients)
  rediverse.directions  ReDiverse-TF direction estimation, files and injection offset
  rediverse.data        training captions, StepSampler, recipe constants, run metadata,
                        prompt files and output paths
  rediverse.claims      atomic per-item claims for multi-worker runs
  rediverse.hub         released adapter resolution and safetensors I/O
  rediverse.video_io    ffmpeg H.264 writer and frame reader

Submodules are imported explicitly (e.g. `from rediverse import seeds`), so the modules that
import without torch (claims, data, hub, video_io) load in environments that lack it.
"""

__version__ = "1.0.0"
