"""ReDiverse on SD3-Medium with the TDM 4-step student (collapsed block: transformer_blocks.12,
image stream).

  sd3.encode_prompts       training-caption and empty-prompt embeddings
  sd3.make_noise_bank      frozen evaluation latents (GenEval-553 x 100)
  sd3.train                ReDiverse-FT adapter
  sd3.convert_weights      trainer checkpoint -> released safetensors
  sd3.estimate_directions  ReDiverse-TF directions
  sd3.generate             teacher / student / rediverse_ft / rediverse_tf / rediverse images
"""
