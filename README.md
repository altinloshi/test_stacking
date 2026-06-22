# test_stacking

Franka 5-cube stacking experiments in IsaacLab — Behavior Cloning and RL policies.

---

## BC LocalTarget Anti-Stall Policy

**Package:** [`bc_localtarget_antistall_9of50/`](bc_localtarget_antistall_9of50/)  
**Full docs:** [`bc_localtarget_antistall_9of50/README_BC_9OF50.md`](bc_localtarget_antistall_9of50/README_BC_9OF50.md)

Target-conditioned Behavior Cloning policy for Franka 5-cube stacking.
Uses LocalTarget representation, RNN checkpoint, anti-stall guard, and planner-forced discrete gripper.

**Main result: 9 / 50 successful rollouts (18%)**

> This is a Stage A / Stage B prototype only. Stage C RL fine-tuning was not completed.

> Videos are intentionally not committed. Evaluation evidence is `.log` files only.
> Git LFS is required to pull `best.pt` and `*.hdf5` dataset files.

### Clone and pull large files

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking
git lfs install
git lfs pull
```

### Verify 9/50 result

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected output:

```
9
```

### Results

| Evaluation | Runs | Successes | Success rate | Successful run IDs |
|------------|------|-----------|--------------|-------------------|
| Main (2026-06-19 17:55:28) | 50 | **9** | **18%** | 4, 5, 9, 22, 31, 33, 36, 43, 49 |
| Sanity (2026-06-19 16:29:50) | 10 | **2** | **20%** | 2, 3 |

See [`bc_localtarget_antistall_9of50/README_BC_9OF50.md`](bc_localtarget_antistall_9of50/README_BC_9OF50.md) for the full reproducibility guide.

---
