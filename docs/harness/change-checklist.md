# Harness Change Checklist

Use this checklist for changes to Harness state, tool exposure, verification,
recovery, stage advancement, completion, or runtime prompts.

- [ ] Update `rpent/harness/lifecycle.py` before changing external state behavior.
- [ ] Add tests for every new legal and illegal transition.
- [ ] Confirm every non-terminal state has an allowed operation.
- [ ] Confirm every lifecycle event is reachable from a declared state.
- [ ] Confirm every advertised session operation has an executable entrypoint.
- [ ] Confirm every advertised transition-causing operation maps to a declared
      lifecycle event from each state where the operation is allowed.
- [ ] Confirm `start_instruction` is a one-shot pre-lifecycle bootstrap: state
      remains `None` until successful plan registration, and repeat starts are
      rejected.
- [ ] Confirm prompts expose `request_exception` before recovery and only
      `apply_recovery` during recovery; do not expose direct scope selection.
- [ ] Keep satisfied-but-not-final outcomes in `verification_required` with the
      dedicated final-verification event; do not report them as uncertain.
- [ ] Preserve immutable outcomes and authoritative completion evidence.
- [ ] Preserve terminal classification and atomic run-status persistence.
- [ ] Regenerate lifecycle documentation with `--write`.
- [ ] Verify generated artifacts with `--check`.
- [ ] Run the focused Harness test suite.
- [ ] Run complete pytest regression.
- [ ] Run a deployed smoke benchmark for behavior changes.
- [ ] Run paired old/new seeds before making success-rate claims.

Generated files are review evidence, not independent specifications. A manual
diagram may be retained for presentations only when it states that
`rpent/harness/lifecycle.py` is authoritative.
