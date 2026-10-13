"""ReDiverse on MiniMax-H3 (teacher) / lightx2v Minimax-h3-Turbo (4-step student).

  h3.config               model ids, pinned revisions, geometry, recipe constants
  h3.model                model loading, text contexts, sampler, teacher rollout, decoder
  h3.split2               two-GPU block split used by ReDiverse-FT training
  h3.encode_prompts       text contexts of the training and VBench prompts
  h3.build_targets        ReDiverse-FT target bank (teacher block-45 video rows at forward 16)
  h3.train                ReDiverse-FT training
  h3.estimate_directions  ReDiverse-TF directions per VBench prompt, or of one --prompt
  h3.generate             the five arms: sample, decode, or a single --prompt
  h3.convert_weights      training checkpoint -> adapter file read by h3.generate --adapter
"""
