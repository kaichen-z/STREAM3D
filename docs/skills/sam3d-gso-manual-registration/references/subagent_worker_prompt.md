# Subagent Worker Prompt

Use this prompt when assigning manual Sim(3) correction work to a subagent.  Fill in the placeholders before sending it.

```text
You are responsible for manually correcting Streaming SAM3D GSO Sim(3) alignment for these scenes only:

{SCENE_LIST}

Repository root:
{REPO_ROOT}

Skill script:
docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py

Adjustment controller:
docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py

State files:
{STATE_FILE_LIST}

Known visual notes from the user or supervisor:
{VISUAL_NOTES}

You are not alone in the codebase.  Other agents may be working on other scenes.  You own only the scratch directories for the scenes listed above.  Do not modify source code, other scene states, accepted scenes, or the original `result.ply` / `result.glb` files.

Supervisor scheduling rule:
At most 5 subagents may be active concurrently across all variants.  If you are assigned work, assume your scene set is intentionally small and do not expand beyond it.

Goal:
Run an actual manual/agent-in-the-loop correction loop until each assigned scene is visually close enough for its reconstruction quality or honestly marked unresolved.  Command completion is not success.  Success means the current `active_sim3` is visually accepted from individual comparison images.  For visibly failed or partial reconstructions, rough acceptance is allowed when semantic direction, scale, and center are broadly correct; document remaining geometry/texture errors as reconstruction limits.

Hard rules:
1. Back up each initial state to `alignment_state.before_manual_adjustment.json` before editing.
2. Do not directly edit raw 4x4 matrices.  Use `sim3_adjustment_controller.py` to propose, render, and apply candidates unless the needed edit cannot be expressed by the controller and you document the exception.
3. Inspect comparison images one at a time.  Start with:
   - `comparisons/000_gt_pred.png`
   - `comparisons/012_gt_pred.png`
   - `comparisons/024_gt_pred.png`
4. Do not use `comparison_contact_sheet.png` as visual evidence for acceptance.  It is only an index for choosing extra individual views.
5. Apply only one visually selected candidate to the canonical `alignment_state.json`; keep `auto_initial_sim3` unchanged.
6. After every applied candidate, rerender:
   `CUDA_VISIBLE_DEVICES={GPU_ID} ./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/manual_register_sam3d_gso.py render --state-file {STATE_FILE}`
7. Reinspect individual comparison images after rerendering.  If the overlay is still wrong, keep iterating or mark the scene unresolved.
8. Write `manual_adjustment_notes.md` in every assigned scene directory.

Suggested correction strategy:
1. Diagnose the failure mode from individual comparison pairs.
2. If `registration_diagnostics.method` starts with `initial` and contains saved candidates, export saved registration candidates first:
   `./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py export-registration-candidates --state-file {STATE_FILE} --round-name registration_candidates --top-k 24`
3. Render those saved candidate triples before inventing new deltas:
   `CUDA_VISIBLE_DEVICES={GPU_ID} ./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py render-candidates --round-dir {ROUND_DIR} --views 0,12,24`
4. Inspect saved candidate `comparisons/000_gt_pred.png`, `012_gt_pred.png`, and `024_gt_pred.png` one candidate at a time.  A lower-ranked candidate may be correct if it fixes front/back, top/bottom, or left/right semantic orientation.
5. If no saved registration candidate fixes the mode, run an orientation candidate round:
   `./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py propose --state-file {STATE_FILE} --round-name orientation_001 --preset orientation`
6. Render candidate triples:
   `CUDA_VISIBLE_DEVICES={GPU_ID} ./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py render-candidates --round-dir {ROUND_DIR} --views 0,12,24`
7. Inspect candidate `comparisons/000_gt_pred.png`, `012_gt_pred.png`, and `024_gt_pred.png` one candidate at a time.  Select the best candidate only if the orientation is visually better.
8. Apply the selected candidate:
   `./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py apply-candidate --state-file {STATE_FILE} --round-dir {ROUND_DIR} --candidate-id {CANDIDATE_ID} --reason "{VISUAL_REASON}"`
9. Rerender the canonical state and reinspect individual comparisons.
10. If orientation is right but residual offset remains, run a micro round with smaller steps:
   `./.env/bin/python docs/skills/sam3d-gso-manual-registration/scripts/sim3_adjustment_controller.py propose --state-file {STATE_FILE} --round-name micro_001 --preset micro --rotation-step-deg 4 --translation-step-fraction 0.01 --scale-step 0.02`
11. Reduce step sizes across rounds rather than making larger unexplained edits.

For each scene, your `manual_adjustment_notes.md` must include:
- initial visual diagnosis from `000`, `012`, and `024`;
- every iteration's hypothesis, controller command, candidate id, and visual reason;
- the render command used;
- the individual comparison files inspected after the final render;
- final status: `accepted`, `accepted_rough`, `improved_not_accepted`, or `not_solved`;
- remaining visible error if not accepted.

Final response format:
For each scene, report:
- status;
- number of edit/render iterations;
- final state path;
- notes path;
- final inspected comparison files;
- concise visual verdict.

Do not claim a scene is accepted just because rendering finished or metrics improved.  If using `accepted_rough`, explicitly state why the remaining mismatch is reconstruction quality rather than fixable global Sim(3).
```
