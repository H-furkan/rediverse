"""ReDiverse on Wan 2.1 T2V-14B (teacher) and the lightx2v 4-step StepDistill-CfgDistill student.

  wan.encode_prompts       UMT5 contexts: training captions, VBench-300, negative prompt
  wan.build_targets        ReDiverse-FT target bank (CFG-distilled teacher, blocks.28, forward 13)
  wan.train                ReDiverse-FT training (torchrun data parallel)
  wan.convert_weights      training checkpoint -> released safetensors adapter
  wan.estimate_directions  ReDiverse-TF teacher directions (base teacher, blocks.28, forward 8)
  wan.generate             the five arms: teacher, student, rediverse_ft, rediverse_tf, rediverse

Shared modules: wan.config (ids, revisions, recipe), wan.model (loading, samplers, decode),
wan.cfgd_teacher (the CFG-distilled target teacher), wan.target_bank (bank I/O).
"""
