"""SD3-Medium models, schedules, the ReDiverse-FT LoRA and the training objective (paper Sec. 4.1).

The teacher is the SD3-Medium transformer. The student is the same transformer carrying the
frozen TDM LoRA at scale 0.125. ReDiverse-FT attaches a second, trainable LoRA to
transformer_blocks 0..12 of the student and minimises

    loss = MSE(h_student, h_teacher)

where h is the image stream of transformer_blocks.12 (out[1] of the MMDiT block; out[0] is the
text stream). h_student is the student's step-1 forward at sigma 1 on the initial noise.
h_teacher is the teacher's conditional forward 9 of its own 28-step schedule, reached from the
same noise and caption by 8 Euler steps guided with CFG 7. Forward 9 reads sigma 0.877382 and its
Euler update lands at 0.857692, the student's sigma after step 1.

Two adapters coexist on the student. diffusers' add_adapter() activates only the new adapter,
and set_adapters() sets requires_grad on every active adapter, so both adapters are re-activated
explicitly and the gradient mask is re-applied after every toggle.

Blocks 13..23 are downstream of the collapsed block, so the student forward and the teacher's hooked
forward are truncated after block 12 (skip_layers); this leaves the loss and its gradient unchanged.
"""
from __future__ import annotations

import re
import time

import torch
import torch.nn.functional as F

from sd3.config import (ANCHOR_BLOCK, D_BLOCK, DTYPE, FT_ADAPTER, LORA_ID, LORA_LAST_BLOCK,
                        LORA_REVISION, LORA_SUFFIXES, LORA_WEIGHT_NAME, MODEL_ID, MODEL_REVISION,
                        N_BLOCKS, N_LORA_MODULES, STEPS, TDM_ADAPTER, TDM_SCALE,
                        TEACHER_GUIDANCE, TEACHER_TARGET_STEP, TEXT_ONLY_AT_ANCHOR)
from rediverse.data import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, REG_LOSS_WEIGHT

# ------------------------------------------------------------------ schedule --

_CACHE: dict = {}


def load_scheduler():
    """The teacher's FlowMatchEulerDiscreteScheduler (shift 3), shared by both models."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler",
                                                           revision=MODEL_REVISION)


def _schedules() -> dict:
    """Timesteps and sigmas of the teacher's 28-step and the student's 4-step schedules."""
    if "sch" not in _CACHE:
        from diffusers import FlowMatchEulerDiscreteScheduler
        base = load_scheduler()
        out = {"shift": float(base.config.shift)}
        for arm, steps in STEPS.items():
            s = FlowMatchEulerDiscreteScheduler.from_config(base.config)
            s.set_timesteps(steps)
            out[arm] = dict(timesteps=[float(t) for t in s.timesteps],
                            sigmas=[float(v) for v in s.sigmas])
        _CACHE["sch"] = out
    return _CACHE["sch"]


def first_timestep() -> float:
    """The step-1 timestep, identical for both schedules (sigma 1)."""
    sch = _schedules()
    for arm in STEPS:
        assert sch[arm]["sigmas"][0] == 1.0, f"{arm}: sigmas[0]={sch[arm]['sigmas'][0]} != 1.0"
    t0 = {sch[arm]["timesteps"][0] for arm in STEPS}
    assert len(t0) == 1, f"schedules disagree on the first timestep: {t0}"
    return t0.pop()


def teacher_rollout_schedule() -> tuple[list[float], list[float]]:
    """(timesteps, sigmas) of the teacher's 28-step schedule; forward i reads sigmas[i] at
    timesteps[i] and its Euler update lands at sigmas[i+1]."""
    sch = _schedules()
    ts = list(sch["teacher"]["timesteps"])
    sg = list(sch["teacher"]["sigmas"])
    assert len(sg) == len(ts) + 1, f"{len(sg)} sigmas for {len(ts)} timesteps"
    assert sg[0] == 1.0 and sg[-1] == 0.0, (sg[0], sg[-1])
    return ts, sg


def teacher_target_sigmas(k: int = TEACHER_TARGET_STEP) -> dict:
    """Noise levels of the k-th teacher forward (1-based): it reads sigmas[k-1] and its Euler
    update lands at sigmas[k]. For k = 9 the landing sigma equals the student's sigmas[1]."""
    ts, sg = teacher_rollout_schedule()
    assert 1 <= k <= len(ts), f"teacher forward {k} out of range 1..{len(ts)}"
    st = _schedules()["student"]["sigmas"]
    s_in, s_out = sg[k - 1], sg[k]
    j_out = st.index(s_out) if s_out in st else None
    if k == 9:
        assert j_out == 1 and round(s_out, 4) == 0.8577 and round(s_in, 4) == 0.8774, (s_in, s_out)
    return dict(k=k, sigma_in=s_in, sigma_out=s_out, t=ts[k - 1],
                student_sigma_index_out=j_out, student_sigmas=list(st))


def premise() -> dict:
    """The step-1 setup: both schedules start at sigma 1 with the same timestep."""
    sch = _schedules()
    t = first_timestep()
    return dict(
        sigma0=1.0, t_used=t, shift=sch["shift"],
        teacher_timesteps=[round(v, 2) for v in sch["teacher"]["timesteps"]],
        student_timesteps=[round(v, 2) for v in sch["student"]["timesteps"]],
        notes=(f"FlowMatchEulerDiscreteScheduler shift={sch['shift']}; sigmas[0] == 1.0 in the "
               f"teacher's {STEPS['teacher']}-step and the student's {STEPS['student']}-step "
               f"schedules and the pipeline feeds raw randn latents, so the step-1 input is the "
               f"initial noise and t={t:.1f} serves both models. The student's step-1 forward is "
               f"conditional (guidance 1); the teacher target rollout uses CFG "
               f"{TEACHER_GUIDANCE} on forwards 1..{TEACHER_TARGET_STEP - 1} and a conditional "
               f"forward {TEACHER_TARGET_STEP}. TDM LoRA scale {TDM_SCALE}."))


# ------------------------------------------------------------------ geometry --

def extract(name: str, out) -> torch.Tensor:
    """The image stream of a JointTransformerBlock output (text, image)."""
    x = out[1] if isinstance(out, tuple) else out
    assert torch.is_tensor(x), f"{name}: image stream is {type(x)}"
    d = int(x.shape[-2]) * int(x.shape[-1])
    assert d == D_BLOCK, f"{name}: D={d} != {D_BLOCK}"
    return x


# -------------------------------------------------------------------- models --

def load_teacher(device: str):
    """The frozen SD3-Medium transformer, bf16, no adapters."""
    from diffusers import SD3Transformer2DModel
    m = SD3Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer",
                                              revision=MODEL_REVISION, torch_dtype=DTYPE)
    assert len(m.transformer_blocks) == N_BLOCKS, len(m.transformer_blocks)
    m.requires_grad_(False).eval()
    return m.to(device, DTYPE)


def load_student(device: str, verbose: bool = True):
    """The 4-step student: the same transformer plus the TDM LoRA at scale 0.125, cast to bf16."""
    from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
    from peft.tuners.tuners_utils import BaseTunerLayer
    m = SD3Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer",
                                              revision=MODEL_REVISION, torch_dtype=DTYPE)
    assert len(m.transformer_blocks) == N_BLOCKS, len(m.transformer_blocks)
    sd = StableDiffusion3Pipeline.lora_state_dict(LORA_ID, weight_name=LORA_WEIGHT_NAME,
                                                  revision=LORA_REVISION)
    StableDiffusion3Pipeline.load_lora_into_transformer(sd, transformer=m, adapter_name=TDM_ADAPTER)
    bound = sum(isinstance(mod, BaseTunerLayer) for mod in m.modules())
    assert bound > 0, "the TDM LoRA bound to no module"
    m.set_adapters([TDM_ADAPTER], weights=[TDM_SCALE])
    m.requires_grad_(False).eval()
    if verbose:
        print(f"[sd3] TDM LoRA bound to {bound} modules at scale {TDM_SCALE}", flush=True)
    return m.to(device, DTYPE)


# ------------------------------------------------------------- ReDiverse-FT --

def lora_target_modules(model) -> list[str]:
    """Full module paths of the ReDiverse-FT LoRA: transformer_blocks 0..12 x LORA_SUFFIXES,
    without the TEXT_ONLY_AT_ANCHOR modules of block 12 (152 modules)."""
    keep = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
    names, existing = [], set()
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) and not hasattr(mod, "base_layer"):
            continue
        m = keep.match(name)
        if not m:
            continue
        block, suffix = int(m.group(1)), m.group(2)
        if block > LORA_LAST_BLOCK or suffix not in LORA_SUFFIXES:
            continue
        existing.add((block, suffix))
        if block == ANCHOR_BLOCK and suffix in TEXT_ONLY_AT_ANCHOR:
            continue
        names.append(name)
    assert len(existing) == (LORA_LAST_BLOCK + 1) * len(LORA_SUFFIXES), len(existing)
    assert len(names) == N_LORA_MODULES, len(names)
    return sorted(names)


def _is_ft_param(param_name: str) -> bool:
    """peft names adapter tensors '<module>.lora_A.<adapter>.weight'."""
    return "lora_" in param_name and f".{FT_ADAPTER}." in param_name


def _is_tdm(param_name: str) -> bool:
    return "lora_" in param_name and f".{TDM_ADAPTER}." in param_name


def apply_grad_mask(student) -> int:
    """Only the ReDiverse-FT adapter trains."""
    n = 0
    for name, p in student.named_parameters():
        want = _is_ft_param(name)
        p.requires_grad_(want)
        n += int(want)
    return n


def adapter_scalings(student) -> dict:
    """{adapter name: sorted distinct per-module scaling}, read off the live modules."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    scales: dict = {}
    for mod in student.modules():
        if isinstance(mod, BaseTunerLayer) and hasattr(mod, "scaling"):
            for k, v in mod.scaling.items():
                scales.setdefault(k, set()).add(round(float(v), 6))
    return {k: sorted(v) for k, v in scales.items()}


def attach_rediverse_lora(student, verbose: bool = True) -> dict:
    """Attach the trainable ReDiverse-FT LoRA (rank 32, alpha 32, B = 0) on top of the frozen
    TDM adapter, keep it in fp32 on the bf16 base, and freeze everything else."""
    from peft import LoraConfig

    targets = lora_target_modules(student)
    student.add_adapter(
        LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                   target_modules=targets, init_lora_weights=True),
        adapter_name=FT_ADAPTER)
    # add_adapter() leaves only the new adapter active; the student needs both.
    student.set_adapters([TDM_ADAPTER, FT_ADAPTER], weights=[TDM_SCALE, 1.0])
    assert set(student.active_adapters()) == {TDM_ADAPTER, FT_ADAPTER}, student.active_adapters()

    # fp32 LoRA on the bf16 base: peft casts the input to lora_A's dtype and the sum back.
    for n, p in student.named_parameters():
        if _is_ft_param(n) and p.dtype != torch.float32:
            p.data = p.data.float()

    apply_grad_mask(student)

    ft_params = [(n, p) for n, p in student.named_parameters() if _is_ft_param(n)]
    tdm = [(n, p) for n, p in student.named_parameters() if _is_tdm(n)]
    assert ft_params, "the ReDiverse-FT adapter created no parameters"
    n_train = sum(1 for _, p in student.named_parameters() if p.requires_grad)
    assert n_train == len(ft_params), (n_train, len(ft_params))
    assert not any(p.requires_grad for _, p in tdm), "the TDM adapter is trainable"
    scales = adapter_scalings(student)
    assert scales.get(FT_ADAPTER) == [1.0], scales.get(FT_ADAPTER)
    assert len(scales.get(TDM_ADAPTER, [])) == 1, scales.get(TDM_ADAPTER)

    info = dict(
        n_target_modules=len(targets),
        n_target_modules_unpruned=(LORA_LAST_BLOCK + 1) * len(LORA_SUFFIXES),
        text_only_at_anchor=list(TEXT_ONLY_AT_ANCHOR),
        n_ft_tensors=len(ft_params), n_ft_params=sum(p.numel() for _, p in ft_params),
        n_tdm_tensors=len(tdm), n_tdm_params=sum(p.numel() for _, p in tdm),
        n_total_params=sum(p.numel() for p in student.parameters()),
        tdm_scaling=scales.get(TDM_ADAPTER), ft_scaling=scales.get(FT_ADAPTER),
        blocks=f"0..{LORA_LAST_BLOCK}", suffixes=list(LORA_SUFFIXES),
    )
    if verbose:
        print(f"[sd3] ReDiverse-FT LoRA: r={LORA_RANK} alpha={LORA_ALPHA} on transformer_blocks "
              f"{info['blocks']} x {len(LORA_SUFFIXES)} projections minus the "
              f"{len(TEXT_ONLY_AT_ANCHOR)} text-only modules of block {ANCHOR_BLOCK} = "
              f"{info['n_target_modules']} modules, {info['n_ft_tensors']} tensors, "
              f"{info['n_ft_params'] / 1e6:.1f}M trainable | frozen TDM: {info['n_tdm_tensors']} "
              f"tensors, {info['n_tdm_params'] / 1e6:.1f}M, scaling {info['tdm_scaling']}", flush=True)
    return info


def load_adapter_state_dict(path: str) -> tuple[dict, dict]:
    """(peft state dict with the adapter-name segment, metadata) from a released
    rediverse_ft.safetensors or a trainer checkpoint (.pth)."""
    if path.endswith(".safetensors"):
        from rediverse.hub import load_lora_safetensors
        sd, meta = load_lora_safetensors(path)
        sd = {k[:-len(".weight")] + f".{FT_ADAPTER}.weight": v for k, v in sd.items()}
        return sd, meta
    ck = torch.load(path, map_location="cpu")
    return ck["state_dict"], {"iteration": str(ck.get("iteration"))}


def load_student_with_adapter(device: str, adapter_path: str, verbose: bool = True):
    """The student with the ReDiverse-FT adapter loaded and both adapters active."""
    m = load_student(device, verbose=verbose)
    attach_rediverse_lora(m, verbose=False)
    sd, _ = load_adapter_state_dict(adapter_path)
    res = m.load_state_dict(sd, strict=False)
    have = {n for n, _ in m.named_parameters()}
    hit = [k for k in sd if k in have]
    nzB = sum(float(sd[k].abs().sum()) for k in sd if "lora_B" in k)
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:5]}"
    assert len(hit) == len(sd) and nzB > 0, f"{len(hit)}/{len(sd)} tensors matched, sum|lora_B|={nzB}"
    own = {n for n, _ in m.named_parameters() if _is_ft_param(n)}
    assert own == set(sd), f"the adapter file covers {len(own & set(sd))}/{len(own)} ReDiverse-FT tensors"
    m.set_adapters([TDM_ADAPTER, FT_ADAPTER], weights=[TDM_SCALE, 1.0])
    if verbose:
        print(f"[sd3] ReDiverse-FT adapter {adapter_path}: {len(hit)}/{len(sd)} tensors matched, "
              f"sum|lora_B|={nzB:.3f}; active={sorted(m.active_adapters())}", flush=True)
    m.requires_grad_(False).eval()
    return m


# ------------------------------------------------------------------ training --

class SD3ReDiverseFT:
    """Teacher, student with the ReDiverse-FT LoRA, the block-12 hooks and the MSE objective."""

    def __init__(self, uncond_path: str, device: str = "cuda", verbose: bool = True):
        self.device = device
        self.t1 = first_timestep()
        self.k_teacher = TEACHER_TARGET_STEP
        self.teacher_guidance = float(TEACHER_GUIDANCE)
        self.teacher_timesteps, self.teacher_sigmas = teacher_rollout_schedule()
        self.target_sigmas = teacher_target_sigmas(self.k_teacher)
        if verbose:
            ts = self.target_sigmas
            print(f"[sd3] teacher target: block {ANCHOR_BLOCK} image stream at teacher forward "
                  f"{self.k_teacher} of {STEPS['teacher']}, sigma_in={ts['sigma_in']:.6f} "
                  f"t={ts['t']:.4f} -> sigma_out={ts['sigma_out']:.6f} == student sigmas[1]", flush=True)
        self.teacher = load_teacher(device)
        self.student = load_student(device, verbose=verbose)

        d = torch.load(uncond_path, map_location="cpu")
        assert d.get("prompt", "") == "", f"{uncond_path} holds prompt {d.get('prompt')!r}, expected ''"
        self._pe_u = d["prompt_embeds"].to(device, DTYPE)
        self._pooled_u = d["pooled_projections"].to(device, DTYPE)
        if verbose:
            print(f"[sd3] teacher rollout guided: CFG {self.teacher_guidance} on forwards "
                  f"1..{self.k_teacher - 1}, conditional hooked forward {self.k_teacher}; "
                  f"uncond = empty prompt {tuple(self._pe_u.shape)}", flush=True)

        self.lora_info = attach_rediverse_lora(self.student, verbose=verbose)
        self.student.enable_gradient_checkpointing()
        self.student.train()
        self.skip_layers = list(range(ANCHOR_BLOCK + 1, N_BLOCKS))

        self._h_s: dict = {}
        self._h_t: dict = {}
        self._capture = False
        self._register_hooks()
        self._graph_checked = False
        self._grad_checked = False
        self._n_micro = 0
        self.t_student = 0.0
        self.t_teacher = 0.0
        self.n_teacher_fwd = 0

    def _register_hooks(self):
        path = f"transformer_blocks.{ANCHOR_BLOCK}"
        s_mod = dict(self.student.named_modules())[path]
        t_mod = dict(self.teacher.named_modules())[path]

        # Gated: under non-reentrant checkpointing the block is replayed during backward.
        def s_hook(_m, _i, out, _p=path):
            if self._capture:
                self._h_s[_p] = extract(_p, out)

        def t_hook(_m, _i, out, _p=path):
            self._h_t[_p] = extract(_p, out)

        s_mod.register_forward_hook(s_hook)
        t_mod.register_forward_hook(t_hook)
        self.site_path = path

    def _fwd(self, model, x, pe, pooled):
        """The step-1 forward at sigma 1, conditional, truncated after the collapsed block."""
        b = x.shape[0]
        t = torch.tensor(self.t1, device=x.device, dtype=torch.float32).expand(b)
        return model(hidden_states=x, timestep=t,
                     encoder_hidden_states=pe.expand(b, -1, -1),
                     pooled_projections=pooled.expand(b, -1),
                     skip_layers=self.skip_layers, return_dict=False)

    @torch.no_grad()
    def teacher_target(self, x, pe, pooled) -> torch.Tensor:
        """The teacher's block-12 image stream at forward tau = 9 of its 28-step trajectory from x.

        Forwards 1..8 are full 24-block forwards with CFG ([uncond, cond] along the batch,
        v = v_u + w (v_c - v_u)) and an Euler update x += (sigmas[i+1] - sigmas[i]) v; forward 9 is
        conditional, batch b, hooked at block 12 and truncated after it."""
        t0 = time.perf_counter()
        self._h_t = {}
        k, b = self.k_teacher, x.shape[0]
        ts, sg = self.teacher_timesteps, self.teacher_sigmas
        pe_b, pooled_b = pe.expand(b, -1, -1), pooled.expand(b, -1)
        pe_u_b, pooled_u_b = self._pe_u.expand(b, -1, -1), self._pooled_u.expand(b, -1)
        xt = x.clone()
        for i in range(k):
            last = i == k - 1
            t = torch.tensor(ts[i], device=x.device, dtype=torch.float32).expand(b)
            if not last:
                v2 = self.teacher(hidden_states=torch.cat([xt, xt]), timestep=torch.cat([t, t]),
                                  encoder_hidden_states=torch.cat([pe_u_b, pe_b]),
                                  pooled_projections=torch.cat([pooled_u_b, pooled_b]),
                                  return_dict=False)[0]
                vu, vc = v2.chunk(2)
                v = vu + self.teacher_guidance * (vc - vu)
                self.n_teacher_fwd += 1          # the doubled batch counts as one extra
            else:
                v = self.teacher(hidden_states=xt, timestep=t,
                                 encoder_hidden_states=pe_b, pooled_projections=pooled_b,
                                 return_dict=False, skip_layers=self.skip_layers)[0]
            assert self.site_path in self._h_t, f"teacher hook did not fire on forward {i + 1}/{k}"
            if not last:
                xt = xt + (sg[i + 1] - sg[i]) * v
            self.n_teacher_fwd += 1
        self.t_teacher += time.perf_counter() - t0
        return self._h_t[self.site_path].detach()

    def student_activation(self, x, pe, pooled) -> torch.Tensor:
        """The student's block-12 image stream at step 1, with the ReDiverse-FT adapter, in the graph."""
        t0 = time.perf_counter()
        self._h_s = {}
        self._capture = True
        try:
            self._fwd(self.student, x, pe, pooled)
        finally:
            self._capture = False
        assert self.site_path in self._h_s, "student hook did not fire"
        self.t_student += time.perf_counter() - t0
        return self._h_s[self.site_path]

    @staticmethod
    def _spread(h: torch.Tensor) -> torch.Tensor:
        """Across-sample std per activation unit (telemetry only)."""
        return (h.float().var(dim=0, unbiased=False) + 1e-6).sqrt()

    def step(self, x, pe, pooled) -> dict:
        """One micro-batch: x [K, 16, 128, 128] paired initial noises of one caption (pe [1, 333,
        4096], pooled [1, 2048]). Returns the loss (in the graph) and detached telemetry."""
        self._n_micro += 1
        if not self._grad_checked:
            n_live = sum(1 for n, p in self.student.named_parameters() if _is_ft_param(n) and p.requires_grad)
            assert n_live > 0, "no ReDiverse-FT parameter requires grad"
            self._grad_checked = True

        h_s = self.student_activation(x, pe, pooled)
        if not self._graph_checked:
            assert h_s.requires_grad and h_s.grad_fn is not None, (
                f"the activation captured at '{self.site_path}' is detached; gradient "
                "checkpointing must be non-reentrant")
            self._graph_checked = True

        h_t = self.teacher_target(x, pe, pooled)
        reg_loss = F.mse_loss(h_s.float(), h_t.float())
        loss = REG_LOSS_WEIGHT * reg_loss

        with torch.no_grad():
            if x.shape[0] >= 2:
                sp_s = self._spread(h_s).mean()
                sp_t = self._spread(h_t).mean()
            else:
                sp_s = sp_t = torch.zeros((), device=h_s.device)
        return {
            "total_loss": loss,
            "reg_loss": reg_loss.detach(),
            "h_std_student": h_s.detach().float().std(),
            "h_std_teacher": h_t.float().std(),
            "spread_student": sp_s,
            "spread_teacher": sp_t,
            "spread_ratio": sp_s / sp_t.clamp_min(1e-12),
        }

    def adapter_parameters(self):
        return [p for n, p in self.student.named_parameters() if _is_ft_param(n)]

    def adapter_state_dict(self) -> dict:
        return {n: p.detach().cpu().clone() for n, p in self.student.named_parameters() if _is_ft_param(n)}

    def timing(self) -> dict:
        """Wall-clock split of one micro-batch (call after torch.cuda.synchronize())."""
        n = max(self._n_micro, 1)
        return dict(n_micro=self._n_micro, teacher_target_step=self.k_teacher,
                    n_teacher_forwards=self.n_teacher_fwd,
                    teacher_forwards_per_micro=self.n_teacher_fwd / n,
                    s_student=self.t_student / n, s_teacher=self.t_teacher / n,
                    s_total_measured=(self.t_student + self.t_teacher) / n)

    def config_dict(self) -> dict:
        return dict(
            anchor_block=ANCHOR_BLOCK, lora_last_block=LORA_LAST_BLOCK,
            lora_target_suffixes=list(LORA_SUFFIXES), text_only_at_anchor=list(TEXT_ONLY_AT_ANCHOR),
            lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            reg_loss_weight=REG_LOSS_WEIGHT, reg_loss_type="mse",
            teacher_target_step=self.k_teacher, teacher_guidance=self.teacher_guidance,
            gradient_checkpointing=True, truncate_after_anchor=True,
            premise=premise(), lora_info=self.lora_info, teacher_target_sigmas=self.target_sigmas,
            image_target=(f"block {ANCHOR_BLOCK} image stream (out[1]) of the frozen teacher at "
                          f"forward {self.k_teacher} of its {STEPS['teacher']}-step FlowMatchEuler "
                          f"schedule, from the same initial noise and caption as the student's step-1 "
                          f"forward: {self.k_teacher - 1} full forwards with CFG "
                          f"{self.teacher_guidance}, then one conditional forward truncated after "
                          f"block {ANCHOR_BLOCK}"),
        )
