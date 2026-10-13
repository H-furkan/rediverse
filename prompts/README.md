# Prompts

Evaluation prompts, the prompt subsets of the paper's tables and the ReDiverse-FT training prompts.
A prompt file holds one prompt per line, and the 0-based line number is the prompt index used in
every file name and seed rule. An index file holds comma-separated prompt indices into a prompt file
(read by `rediverse.data.parse_ids`, which also accepts `a-b` ranges).

| File | Content | Used for |
|---|---|---|
| `geneval_553.txt` | the 553 GenEval prompts: the `prompt` field of every entry of GenEval's evaluation metadata file, in file order | image generation and evaluation (`<idx:03d>_s<seed>.png`) |
| `geneval_subset100.txt` | 100 indices into `geneval_553.txt`: `round(linspace(0, 552, 98))` together with 27 and 62 | the four TPIPS aspects of the image tables |
| `geneval_subset50.txt` | the first 50 indices of `geneval_subset100.txt` | the SD3 ablation table and the design-choice tables (supplementary) |
| `h3_train_geneval_ids.txt` | 20 indices into `geneval_553.txt` | MiniMax-H3 ReDiverse-FT training prompts |
| `vbench_300.txt` | 300 VBench prompts | video generation and evaluation (`<prompt>-<k>.mp4`) |
| `vbench_subset100.txt` | indices 0, 3, 6, ..., 297 into `vbench_300.txt` | the four TPIPS aspects of the video tables |
| `vbench_subset30.txt` | indices 0, 10, 20, ..., 290 into `vbench_300.txt` | camera-trajectory diversity (Cam) |
| `download_train_captions.py` | writes `train_captions.txt`, the 100 ReDiverse-FT training captions | ReDiverse-FT training on SDXL, SD3 and Wan 2.1 |

`vbench_300.txt` is a selection from the prompt lists of the VBench suite (its per-dimension lists
and its list of all prompts): each of the 11 prompt dimensions contributes a quota proportional to
its size, taken at evenly spaced positions, and the rest is filled from the list of all prompts. The
prompts themselves are copied unchanged.

The training captions are not stored in the repository: `download_train_captions.py` takes the first
100 lines of `data/diffusiondb.txt` of the [DP-DMD](https://github.com/Multimedia-Analytics-Laboratory/dpdmd)
repository at a pinned commit (prompts of [DiffusionDB](https://github.com/poloclub/diffusiondb),
CC0 1.0) and checks both the downloaded file and the result by SHA-256. Caption `ci` is line `ci`
(0-99) of `train_captions.txt`.

## Licenses

`geneval_553.txt` reproduces the prompts of [GenEval](https://github.com/djghosh13/geneval), which is
released under the MIT License:

```
MIT License

Copyright (c) 2023 Dhruba Ghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`vbench_300.txt` is a selection of the prompts of [VBench](https://github.com/Vchitect/VBench), which
is licensed under the Apache License, Version 2.0. A copy of that license is the repository's
[`LICENSE`](../LICENSE).

The index files are part of this repository and are released under its Apache License 2.0.
