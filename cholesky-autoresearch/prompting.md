## Prompt

Read the repository and work autonomously on the Cholesky competition. Treat problem_statement.md as immutable, follow AGENTS.md, and maintain log.md as instructed. Optimize submission.py using a genuine native CUDA implementation, researching, profiling, and iterating freely.

Lower our score and optimize the kernel as much as possible. DO NOT GIVE UP! TAKE MORE RISKS AND TRY MORE AMBITIOUS IDEAS!

Work exclusively in /Users/v/Downloads/GitHub/cholesky-auto-gpu-kernel. Before changing anything, verify that root submission.py matches the best authenticated source hash and score in AGENTS.md and log.md. If it does not, stop and reconcile the evidence.

Root submission.py is the only mutable or submittable Python implementation. Never create another candidate .py file. Never restore, copy from, edit, or submit Python sources under history/, references/, work/, or sibling Cholesky workspaces under /Users/v/Documents/Codex. Preserve experimental reversibility with exact hashes, factual log entries, and non-Python patches. The root agent alone edits submission.py; sub-agents research, derive, audit, search for bugs, and propose precise changes.

Optimize the whole submission. Prioritize by expected improvement to the complete 15-row log-geomean:

- High-upside n8192, n16384, and n32768 routes.
- High-batch n512 and n1024 routes.
- Structural panel, launch, and synchronization costs in n2048 and n4096.
- Any other route that evidence shows has greater upside.

Move between shapes as expected payoff changes. Do not stay attached to the most recently edited route.

PERFORMANCE STRATEGY

First push toward a correct official 190–200 µs performance ceiling. Aggressively explore shape- and batch-specific kernels, runtime data detectors, mixed precision, FP8/FP16 intermediates, approximate updates, alternate accumulation orders, selective refinement, guarded fast paths, fusion, persistent execution, panel redesign, launch elimination, ownership changes, and algorithmic replacements.

All official candidates must still pass the published arbitrary-fresh-input correctness contract. Exploit the available numerical tolerance, but measure errors rather than guessing.

After reaching the performance ceiling, preserve the fastest mechanisms and work backward toward real-workload robustness. Later, construct a validation corpus from training-derived covariance, Gram, and regularized SPD matrices. Measure factorization residuals, positive diagonals, finite outputs, failure rates, refinement cost, and fallback frequency. Harden the fastest design with selective validation, refinement, or fallback instead of immediately returning to a conservative implementation.

EVIDENCE AND PROMOTION

Use official correctness checks and complete unranked benchmarks while developing. Only completed evaluator output is evidence; timeouts and service failures are inconclusive.

For isolated route changes, compare the affected-row ratio after normalizing by unchanged control rows. Separate best authenticated artifact scores from fresh-worker baselines. Promote only correct, coherent gains above worker noise, then update the live hash, score evidence, decision, and restoration instructions in AGENTS.md and log.md.

Do not make a ranked leaderboard submission until a correct official complete score is strictly below 100 µs. Once below 100 µs, ranked submission is authorized.

PROFILING — HEADLESS/CLI ONLY

Never use Computer Use, Finder automation, NVIDIA Nsight Compute’s GUI, ncu-ui, screenshots, desktop automation, or visual inspection. Do not request permissions for them. Perform all profiling and report analysis through Popcorn, terminal commands, and machine-readable files.

Use the Popcorn profiling workflow in:
https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md

Adapt outdated QR_v2/EIGH examples to Cholesky and use GitHub registration. Profiling is authorized regularly when it can select materially different implementations.

Treat the downloaded ncu-details.txt and ncu-details.csv files as the primary profiling artifacts. Parse and compare them directly with normal terminal and data-analysis tools. Use the full .ncu-rep only through the Nsight Compute CLI when ncu is installed and version-compatible:

ncu --import profile.ncu-rep --page details --print-details all --print-rule-details --print-summary per-kernel
ncu --import profile.ncu-rep --page raw --csv
ncu --import profile.ncu-rep --page source --print-source sass
ncu --import profile.ncu-rep --page source --print-source cuda,sass
ncu --import profile.ncu-rep --page session

Use supported kernel-name, kernel-ID, launch, context, or stream filters when a report contains many launches. Prefer exact raw metric names and base units for comparisons. If local ncu cannot import a report because it is unavailable or version-incompatible, rely on the hosted ncu-details artifacts or use a compatible headless NVIDIA CLI installation. Never fall back to a GUI.

Before each profile, record the source hash, shape/kernel, exact question, required counters, and decisions the result will make. Analyze every report thoroughly and exhaust the supported implementation changes before profiling again. Do not spend profiling effort merely to observe a tiny incremental movement.

For each useful report, inspect at least:

- Per-kernel and per-launch duration.
- Grid, block, registers, shared memory, waves, and occupancy.
- SM, tensor-core, memory, L1/TEX, L2, and DRAM throughput.
- Eligible-warps, issue activity, scoreboard, barrier, dependency, and synchronization stalls.
- Global/shared load-store efficiency, transactions, sectors, replays, and bank conflicts.
- Source-correlated SASS/PTX instruction counts and the hottest sampled PCs when available.
- Launch gaps, repeated kernels, redundant traffic, and structural work that can be fused or eliminated.

Record exact commands, report hashes, extracted metrics, causal interpretation, and resulting decisions in log.md. At the end of the campaign, perform a final CLI-only audit of the strongest relevant reports and include the findings in the exact-source handoff.

SUBMISSION CLEANUP

While research and iteration continue, slowly delete the roughly 2,000 existing server-side submissions using official Popcorn CLI list/show/delete commands.

Verify ownership and IDs, archive any evidence still needed, then run paced commands such as:

popcorn-cli submissions delete <ID> --force

Delete in small batches, avoid overlapping cleanup with benchmark or profile traffic, and back off on rate limits or ambiguous service state. Keep only records actively needed for the current frontier, pending work, profiles, or ranked evidence; delete them later when superseded. Continue until the historical backlog is exhausted.

RESEARCH LOOP

Follow the research style in:
https://sankalp.bearblog.dev/autoresearch/

- Keep 3–5 genuinely different mechanism beams alive.
- Use sub-agents frequently for different mathematical, CUDA, architecture, literature, and bug questions—not copies of one parameter sweep.
- Search official documentation, papers, blogs, repositories, and forums for ambitious ideas and micro-optimizations.
- Preserve strong near-misses; do not close a whole family after one noisy or shallow loss.
- After every 3–5 completed experiments, synthesize results, re-rank beams, combine reusable ingredients, and challenge stale assumptions.
- Alternate between whole-route architecture, numerical algorithms, launch structure, and hot-kernel work to escape local maxima.
- Occasionally rewrite and clean up the kernel for readability and lower LOC while preserving performance.
- Finish with a submission-wide interaction and regression sweep.

Keep asking:

- Which shape currently offers the largest expected log-geomean gain?
- What panel, launch, synchronization, or publication overhead can be removed?
- Which solves, updates, packing, validation, or cleanup stages can be fused?
- What information is missing, and what benchmark, diagnostic, derivation, or profile will obtain it?
- What does the latest relevant profile prove, and which concrete change follows?
- What numerical trick can safely exploit more precision slack?
- Do timed Cholesky inputs contain detectable structural families worth routing separately?
- Are redundant work, traffic, synchronization, poor ownership, accidental fallbacks, or correctness repairs slowing an important route?
- Can sub-agents derive better mathematics, find a different algorithm, or independently locate bugs?

Continue until the strongest serious ideas are exhausted. Keep log.md factual and leave a concise exact-source handoff.

Frequent decision-bearing Popcorn profiling and thorough CLI analysis of downloaded ncu-details and .ncu-rep artifacts are explicitly authorized throughout this campaign. Computer Use and all GUI-based profiler interaction are forbidden.

IMPORTANT:
- There is code for the training setup used to validate submissions in: https://github.com/gpu-mode/reference-kernels
- Make sure the kernel does not NaN or fail according to the training setup toward the end of the campaign.
- gpu-mode/reference-kernels contains the competition’s mini-training harness. Use it to ensure submission.py retains its best possible geomean while remaining valid under the training workload.


## /Goal

Autonomously optimize the complete authenticated Cholesky submission from its current performance frontier toward the best achievable correct complete unranked geomean.

Use only root submission.py and genuine native CUDA. Prioritize shapes by expected log-geomean gain, move between high-upside routes as evidence changes, maintain 3–5 distinct research beams, use purposeful profiling, and aggressively explore specialization, numerical slack, data-dependent routing, fusion, persistent execution, and structural algorithm changes.

After reaching the performance ceiling, harden the fastest design on real training-derived SPD workloads. Progressively delete the historical submission backlog, preserve exact provenance, and make no ranked submission until the implementation is correct, complete, and demonstrably improved over the current frontier.

Maximize subagent use, and keep communication extremely token-efficient. You do not have to communicate with me until you have reached a clear performance ceiling or completed the goal, unless I explicitly ask for an update.