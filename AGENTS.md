# Repository agent instructions

These instructions apply to the entire repository.

## Default workflow

- Build a small, runnable MVP directly. Prefer working code over process artifacts.
- Do not use Superpowers skills, brainstorming workflows, visual-companion sessions, or
  `.superpowers/` artifacts in this repository.
- Do not create design specs or implementation plans unless the user explicitly asks for them.
- Do not introduce multi-agent implementation or multi-round review workflows unless the user
  explicitly requests delegation or a detailed review.
- Keep changes focused: use few files, reuse existing interfaces, and avoid speculative abstractions,
  plugin systems, or framework rewrites.
- Make reasonable low-risk assumptions and continue instead of asking a sequence of multiple-choice
  questions.

## Delivery

- Run the smallest relevant tests while developing. Run broader checks once before delivery when the
  change warrants them.
- Fix confirmed critical issues directly; do not add lengthy approval or review loops.
- Keep user-facing updates short and outcome-oriented.
- After completing and validating an implementation task, commit it, push the branch, and open a
  pull request by default unless the user explicitly asks not to. Do not add unrelated changes.
- Preserve existing user files and untracked run artifacts.

## Project boundaries

- This project is for research, backtesting, and paper trading only. Do not add live broker or real
  money execution unless the user explicitly changes that scope.
- Keep LLM calls bounded and visible. Prefer MiniMax China-region models or local Codex according to
  the existing provider interfaces.
- Never persist API keys, raw credentials, or hidden model reasoning.

Explicit user instructions always override these defaults.
