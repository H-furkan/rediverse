# ReDiverse files added to FastGen

This checkout is NVlabs/FastGen at commit `060b8570a62c105c1a8a4f8b6f2e8089a3f19666`
(Apache License 2.0, see `LICENSE` and `licenses/`), prepared by `sdxl/setup_fastgen.sh` of the
ReDiverse repository. No upstream file is modified. The following files were added by ReDiverse
(Apache License 2.0):

| File | Purpose |
|---|---|
| `fastgen/methods/rediverse_ft.py` | `ReDiverseFTModel`: step-1 LoRA on the DMD2 student, activation-matching loss against the teacher's forward-7 activation |
| `fastgen/datasets/rediverse_ft.py` | `ReDiverseFTLoader`: one training caption with 8 seeded noises per micro-batch |
| `fastgen/configs/experiments/SDXL/config_rediverse_ft.py` | the training recipe of the released SDXL adapter |
| `fastgen/callbacks/loss_print.py` | scalar loss lines in the text log |

The added modules import `rediverse` and `sdxl` from the ReDiverse repository root, which
`python -m sdxl.train` puts on `PYTHONPATH`.
