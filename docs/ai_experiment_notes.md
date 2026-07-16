# AI Experiment Notes

This note freezes the current learned-policy experiment state. It records the best model configuration so far, what improved, what failed, and what should be tried next.

## Current Best Configuration

- Experiment directory: `experiments\5_19_bad_w010_n1_h128_b4_e5_s50`
- Feature version: v2
- Model: candidate MLP, hidden size 128
- Training: batch size 4, epochs 5, seed 123, CPU
- Progress auxiliary loss: 0.1
- Comparison data: `heuristic_bad` negatives, one negative per sample
- Comparison loss weight: 0.1
- Direct play: 1 win out of 50 seeds
- Average home cards: 5.44
- Verified win: seed 32
- Seed 32 victory trace was exported and replay-verified.

## Positive Findings

- Feature v2 improved direct-play home-card progress compared with the first learned-policy feature set.
- `heuristic_bad` negative sampling was more useful than random comparison negatives for direct play.
- The progress auxiliary loss gave a small improvement in average home-card progress.
- Saving won traces made the seed 32 learned-policy victory reproducible and auditable with `replay.py`.

## Negative Findings

- Re-injecting a single learned win trace degraded direct-play behavior instead of improving it.
- Random comparison loss improved offline dataset accuracy, but did not improve direct-play performance.
- Loop comparison data mostly shifted the failure shape from `loop_detected` toward `no_legal_moves`; it did not increase wins.
- Structural-degradation comparison data improved offline dataset accuracy, but damaged direct play and lost the seed 32 win.

## Current Failure Profile

- `no_legal_moves` failures show more buried low cards and lower movable suffix totals.
- `loop_detected` terminal choices are mostly `COL_TO_COL` and `FREE_TO_COL`.
- The latest failure diagnostics show very low remaining buffer slots near failures.
- Many chosen actions still reduce buffer capacity, even when they do not release low cards or move cards home.

## Next Recommendations

- Do not keep sweeping comparison loss weights for the current datasets.
- Do not keep repeating a single learned win trace as augmentation.
- Prefer a stronger state-value or progress target next, or derive more reliable preferences from solver search trees rather than terminal failure heuristics alone.
- If a GPU environment is available, expand the seed dataset before increasing model complexity.
- Keep the model small until the training target is more predictive of direct-play progress.

## Not Recommended Next

- Do not connect learned policy playback to the GUI yet.
- Do not add reinforcement learning or self-play before the supervised/value data contract is clearer.
- Do not commit generated models, traces, JSONL datasets, reports, or experiment directories.
