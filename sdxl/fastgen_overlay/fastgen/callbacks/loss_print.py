# SPDX-License-Identifier: Apache-2.0
# Added by ReDiverse on top of NVlabs/FastGen (commit 060b857); not part of upstream FastGen.
# Scalar loss lines in the text log (FastGen's own scalar logging goes to Weights & Biases).
from fastgen.callbacks.callback import Callback
import fastgen.utils.logging_utils as logger


class LossPrintCallback(Callback):
    def __init__(self, every_n: int = 25):
        super().__init__()
        self.every_n = every_n

    def on_training_step_end(self, model=None, data_batch=None, output_batch=None,
                             loss_dict=None, iteration=0, **kwargs):
        if iteration % self.every_n == 0 and loss_dict:
            vals = {k: (f"{v.item():.5f}" if hasattr(v, "item") else str(v))
                    for k, v in loss_dict.items()}
            logger.info(f"LOSS iter={iteration} " + " ".join(f"{k}={v}" for k, v in vals.items()))
