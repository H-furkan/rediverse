"""Merge the n outputs of evaluation.image.run_metrics --shard i --num-shards n into one metrics file.

  python -m evaluation.image.merge_shards --out <set>/metrics_gpu.json <set>/metrics_gpu_s0of4.json ... (all n shards)

Shard i holds the prompt groups with list index k % n == i (ascending k); per-image lists are
restored to the global prompt order, and per-prompt means + the std over prompts are added.
"""
import argparse
import json
import statistics


def merge(parts):
    n = len(parts)
    for p in parts:
        sn = int(p.get("shard", "0/1").split("/")[1])
        assert sn == n, f"expected {n} shards, file says {p.get('shard')}"
    parts = sorted(parts, key=lambda p: int(p["shard"].split("/")[0]))
    assert [int(p["shard"].split("/")[0]) for p in parts] == list(range(n)), "shard indices must be 0..n-1"
    R = parts[0]["num_images"] // parts[0]["num_prompts"]
    total_prompts = sum(p["num_prompts"] for p in parts)
    out = {"images_dir": parts[0]["images_dir"], "prompts": parts[0]["prompts"], "num_prompts": total_prompts,
           "num_images": sum(p["num_images"] for p in parts), "metrics": {}}
    if all("prompt_ids" in p for p in parts):
        ids = [None] * total_prompts
        for p in parts:
            si = int(p["shard"].split("/")[0])
            for j, v in enumerate(p["prompt_ids"]):
                ids[si + n * j] = v
        out["prompt_ids"] = ids
    for name in parts[0]["metrics"]:
        per_prompt = [None] * total_prompts; per_image = [None] * total_prompts
        for p in parts:
            si = int(p["shard"].split("/")[0]); m = p["metrics"][name]
            if "per_image" in m:
                chunks = [m["per_image"][j * R:(j + 1) * R] for j in range(p["num_prompts"])]
                for j, ch in enumerate(chunks):
                    k = si + n * j; per_image[k] = ch; per_prompt[k] = sum(ch) / len(ch)
            elif "per_prompt" in m:
                for j, v in enumerate(m["per_prompt"]):
                    k = si + n * j; per_prompt[k] = v
        assert all(v is not None for v in per_prompt), f"{name}: missing prompts after merge"
        flat = [x for ch in per_image for x in (ch or [])]
        res = {"mean": (sum(flat) / len(flat)) if flat else sum(per_prompt) / len(per_prompt),
               "per_prompt": per_prompt, "std_over_prompts": statistics.pstdev(per_prompt), "n_prompts": total_prompts,
               "seconds": sum(p["metrics"][name].get("seconds", 0) for p in parts)}
        if flat:
            res["per_image"] = flat
        out["metrics"][name] = res
        print(f"[merge] {name}: mean={res['mean']:.4f} std_over_prompts={res['std_over_prompts']:.4f} n={total_prompts}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    out = merge([json.load(open(f)) for f in a.shards])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("[merge] wrote", a.out)


if __name__ == "__main__":
    main()
