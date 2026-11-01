# szeinali.github.io

Personal site with a profile and a live project log. Pages are marked `noindex, nofollow, noarchive`, so search engines are asked not to list or cache them. Anyone with the link can still open them.

```
index.html                       profile
projects/sim2real/index.html     project log page
projects/sim2real/entries.json   the log: one entry per milestone, newest shown first
projects/sim2real/data/<entry>/  metrics.json, distributions.json, rollouts.json
assets/viz.js                    renders every visual listed in entries.json (no dependencies)
code/sim2real/                   scripts that produce the data
```

## Add a new result
1. Run a script with `--out projects/sim2real/data/<new-entry>`, e.g.
   `python code/sim2real/train_mujoco_reacher.py --algo sac --out projects/sim2real/data/week1-mujoco`
2. Copy an entry in `entries.json`, point its `src` paths at the new folder, and write the title, takeaways and a `why` for each visual.
3. Commit and push. GitHub Pages redeploys automatically.

Visual types: `kpis`, `learning_curve` (`metric`: `success_rate` | `mean_return` | `mean_final_dist_cm`), `dist_over_time`, `histogram`, `arm_replay`.

Local preview: `python -m http.server` in the repo root, then open http://localhost:8000.
