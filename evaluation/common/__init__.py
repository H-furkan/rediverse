"""Helpers shared by the image and video evaluation.

  atomic         complete-or-absent writes of cache and metric files
  image_sets     <prompt_idx:03d>_s<seed>.png sample sets: grouping by prompt with a strict
                 prompts x seeds grid check
  pairwise       mean pairwise cosine over a set of embeddings (the diversity reduction)
  tpips_factors  the pinned TPIPS-Embed checkpoint and the five factor texts (overall + four aspects)
"""
