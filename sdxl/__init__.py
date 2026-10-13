"""ReDiverse on SDXL with the DMD2 4-step student (collapsed block: up_blocks.0.attentions.0).

  sdxl.make_noise_bank      frozen evaluation latents (GenEval-553 x 100)
  sdxl.encode_captions      training-caption embeddings
  sdxl.build_init_ckpt      FastGen initialisation checkpoint (fp32 DMD2 UNet)
  sdxl.train                ReDiverse-FT adapter (FastGen + sdxl/fastgen_overlay)
  sdxl.extract_adapter      FastGen checkpoint -> adapters/rediverse_ft_<iteration>.pth
  sdxl.convert_weights      rediverse_ft_<iteration>.pth -> released safetensors
  sdxl.estimate_directions  ReDiverse-TF directions
  sdxl.generate             teacher / student / rediverse_ft / rediverse_tf / rediverse images

Shared modules: sdxl.config (ids, revisions, recipe), sdxl.teacher_rollout (the teacher's Euler
trajectory hooked at the collapsed block).
"""
