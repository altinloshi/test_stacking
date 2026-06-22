# test_stacking

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking
git lfs install
git lfs pull
```

```bash
ls -lh bc_localtarget_antistall_9of50/policy/best.pt
head -5 bc_localtarget_antistall_9of50/policy/best.pt
```

```bash
find bc_localtarget_antistall_9of50/datasets -type f \( -name "*.hdf5" -o -name "*.h5" \) -exec ls -lh {} \;
```

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=False" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

```bash
cd /path/to/test_stacking
python -m pip install -e source/franka_5cube_stack
```

```bash
cd /path/to/IsaacLab

./isaaclab.sh -p /path/to/test_stacking/bc_localtarget_antistall_9of50/scripts/custom/bc_detector_expert/play_bc_rnn_target_conditioned.py \
  --base_script /path/to/test_stacking/bc_localtarget_antistall_9of50/scripts/custom/run_5cube_camera_hierarchical_success_fixed.py \
  --checkpoint /path/to/test_stacking/bc_localtarget_antistall_9of50/policy/best.pt \
  --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --max_steps 700 \
  --enable_cameras \
  --disable_yaw_grasp \
  --device cuda:0
```

```bash
cd /path/to/test_stacking
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh
```

```bash
LATEST=$(ls -td /tmp/localtarget_antistall_50x_eval_* | head -1)

echo "$LATEST"

grep -R "FINAL PHYSICAL STACK SUCCESS=True" "$LATEST"/run_*.log | wc -l
grep -R "FINAL PHYSICAL STACK SUCCESS=False" "$LATEST"/run_*.log | wc -l
```
