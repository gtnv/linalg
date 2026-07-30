# Cholesky Agent Notes

This repository holds one native CUDA submission and one research ledger. Follow these rules unless the user explicitly overrides them.

## Repository contract

- Work only in `/Users/v/Downloads/GitHub/cholesky-auto-gpu-kernel`.
- Read `problem_statement.md`, `submission.py`, and relevant `log.md` entries before changing code.
- Treat `problem_statement.md` as immutable.
- Root `submission.py` is the only mutable or submittable Python implementation.
- Never create another candidate `.py` file.
- Keep the repository limited to `problem_statement.md`, `AGENTS.md`, `submission.py`, and `log.md`.
- Do not use Git unless the user explicitly asks.
- The root agent alone edits `submission.py`. Sub-agents research, derive, audit, find bugs, and propose precise changes.

## Source integrity

- Keep the complete operational implementation inside `submission.py`.
- Keep authored Python, C++, CUDA, and required CUTLASS source visible as ordinary text.
- Do not hide source in compressed payloads, encoded blobs, archives, generated packages, external downloads, or precompiled binaries.
- Literal source may be written to temporary compiler inputs at runtime. `submission.py` remains the sole source of truth.
- Preserve required third-party licenses and keep vendor source clearly separate from authored code.
- The implementation must perform genuine Cholesky factorization for arbitrary fresh inputs allowed by the contract.
- Never exploit seeds, call order, tensor identity, reused addresses, evaluator state, or timing behavior.
- Remove code only after ruling out every live dispatch, build, launch, synchronization, and resource-lifetime effect.
- Keep support code only when it builds, launches, synchronizes, or returns the implementation.

## Code style

- Format authored C++ and CUDA conventionally and consistently.
- Do not mechanically reformat embedded vendor source.
- Put comments beside the shape, route, launch, precision, memory, or synchronization choice they explain.
- Comments should be short, direct, and intuitive. Informal grammar is fine.
- Explain the optimization or constraint. Do not narrate the development process.
- No docstrings, banner essays, source-note headings, restoration narration, or gratuitous blank-line runs.
- Prefer simple control flow and compile-time specialization over runtime indirection when behavior is fixed.

## `log.md` is the history

- `log.md` is the only research history, evidence ledger, and writeup source.
- Do not create history, work, reference, profile, report, patch, or proof directories.
- Do not retain raw build, benchmark, submission, or profiler artifacts in the repository.
- Use temporary storage outside the repository when an analysis needs intermediate files.
- Before deleting useful evidence, record its exact command, source identity, relevant output, interpretation, and decision in `log.md`.
- Record each meaningful attempt with:
  - hypothesis and targeted cost;
  - affected shapes, routes, and functions;
  - exact source digest;
  - implementation change;
  - correctness evidence;
  - per-shape timing and whole-table result when available;
  - profiler counters when used;
  - keep, reject, combine, or revisit decision;
  - enough detail to reverse the edit manually.
- Keep entries factual and chronological.
- Rewrite stale summaries when they conflict with later evidence, but preserve the underlying attempt record.
- Distinguish measured results, static reasoning, numerical analysis, and untested expectations.

## Evidence discipline

- Only completed evaluator output is correctness or performance evidence.
- Timeouts, service errors, and incomplete runs are inconclusive.
- Static inspection proves source structure, not runtime correctness or speed.
- Never attach one source's timing to different source bytes.
- Compare isolated route changes using affected rows and stable control rows.
- Treat raw worker variation as noise until controls support a causal conclusion.
- Promote only candidates that pass the required correctness checks and show a coherent gain beyond noise.
- Prefer equal performance with simpler code over added complexity.
- Run the published reference-kernels training workload before a final handoff when the environment permits it.
- Never make a ranked or leaderboard submission without explicit user authorization.

## Research loop

- Do not optimize as a single-incumbent hill climb.
- Keep three to five genuinely different mechanism beams alive when useful:
  - a low-risk improvement near the strongest design;
  - a promising near-miss;
  - a structural or algorithmic change;
  - a shape-specific route;
  - a cleanup or compiler-specialization beam when evidence supports it.
- Use sub-agents for different mathematical, CUDA, architecture, literature, and bug questions. Do not duplicate one shallow sweep.
- Search official documentation for contracts. Use papers, blogs, repositories, and forums for hypotheses. Use hardware evidence for decisions.
- Preserve useful near-misses in `log.md`. Do not preserve alternate source files.
- Do not kill a family after one noisy or poorly tuned result.
- After every few experiments, rerank the beams, combine reusable ingredients, and challenge stale assumptions.
- Take ambitious ideas seriously, but isolate their correctness and performance effects.
- Keep asking:
  - What panel, launch, publication, or synchronization work can disappear?
  - Which stages can fuse while data is already resident?
  - Can one pass compute several reductions?
  - Which runtime choices can become compile-time constants?
  - What do compiled PTX and SASS reveal?
  - Where can lower precision safely use the checker's numerical slack?
  - Which shape offers the largest whole-table payoff?
  - What information is missing, and what smallest experiment will obtain it?
  - Is redundant traffic, fallback work, repair work, or accidental serialization still active?

## Headless profiling

- Profile only when the result can choose between materially different implementations or close a beam.
- Use Popcorn and terminal tools only.
- Never use Computer Use, Finder automation, screenshots, `ncu-ui`, or any profiler GUI.
- Before profiling, record in `log.md`:
  - source identity;
  - shape and kernel;
  - exact question;
  - required counters;
  - decisions each outcome would trigger.
- Inspect per-launch duration, launch gaps, grid and block shape, registers, shared memory, occupancy, waves, tensor activity, memory traffic, cache behavior, stalls, synchronization, bank conflicts, and source-correlated PTX or SASS.
- Prefer exact raw counter names and base units.
- Record the command, report digest, decisive metrics, causal interpretation, and resulting action in `log.md`.
- Remove downloaded reports and profiler artifacts after their necessary evidence is recorded.
- Do not profile merely to observe a tiny change that a normal benchmark can decide faster.
