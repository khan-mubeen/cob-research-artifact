# Frozen experimental framing (recorded before Phi/Gemma training)

2026-07-20. This freezes how results are reported so the hybrid guard can never
be silently substituted for pure-SLM performance.

## Primary research arms (the architecture comparison)

1. **rule-based** — deterministic `decide_*`, the label oracle.
2. **base SLM** — the untuned checkpoint.
3. **pure Protocol v2 fine-tuned SLM** — one v2 run per architecture, equal
   pre-registered budget (`protocol_v2.json`), NO guard, NO post-processing.

The architecture ranking is decided on arm 3 only. Every model gets exactly one
Protocol v2 run at the same budget; no per-model hyperparameter tuning.

## Additional engineering arm (not part of the architecture ranking)

4. **pure Protocol v2 fine-tuned SLM + Guard B** — the deterministic dynamic
   feasibility guard (`apply_dynamic_guard`), disabled by default, applied after
   `slm_decide_*` and before publishing.

Rules for arm 4, enforced in every report:
- It is labelled a **hybrid engineering configuration**, never presented as the
  pure-SLM result.
- Guard B **only suppresses** a dynamically-infeasible SLM BID. It **cannot
  recover** a false NO_BID, and it **never** computes or replaces a price.
- Protocol v2 robustness **failures remain failures**. The pre-registered
  ≥9/12-per-class bar is not softened retrospectively. A guarded arm that passes
  a boundary metric the pure arm failed is reported as the guard's contribution,
  explicitly.
- False BIDs removed and false NO_BIDs remaining are reported **separately**.

## Why the guard is defensible for current v4 deployment (engineering exception)

The exhaustive real-v4 census contains **no eligible tailoring pairs near the
boundary** (max eligible ratio 0.608; `audit_guards`/decision census). On real
traffic the guard therefore almost never fires, and when it does it is provably
correct (it equals the rule). The 329,313-pair decision census shows Guard B's
recompute is byte-identical to the rule decision; the guard's only production
effect is to catch a hypothetical off-distribution infeasible BID. It is
recommended for deployment robustness while the **research** result stays pure.

## Guard ≡ rule equivalence (used when scoring the guarded arm from saved evals)

Guard B's verdict is, by construction and by the 329k-pair regression, identical
to the rule decision for any (agent, order). On the rule-labelled evaluation
sets every NO_BID label is a dynamic reason and static eligibility passes, so
the guarded decision for a row equals: NO_BID if the row's rule label is NO_BID
and the model bid, else the model's own decision. The guarded-arm metrics are
computed this way and are exact, not approximate.
