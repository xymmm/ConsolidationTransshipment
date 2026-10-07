# Monotone approximation report

## model3d_Cf8_N400_T5_lam15_lam23_h1_cu1_pi16_pi26_e039e6  (3-D model, Cf = 8)
- Validity: re-evaluated optimal table vs solver V* differs by 0.00 -> OK.
- Optimal table: 331 structural I₂-violations (present under both tie rules; 331 with ties to wait, 331 with ties to dispatch). The optimal policy is NOT monotone in I₂.
- Other diagnostics: τ-violations 2, rows with a non-upper dispatch set (b₁ ≤ cap) 0, exact-tie cells 13.
- fill M⁻: changed 690 cells in 331 periods. Monotonicity (b̄₁ non-increasing in I₂) is established; no τ-direction violations afterwards.
  Reach: in the worst period b̄₁ was moved at 34 I₂ levels, by up to 1 units.
  CAUTION: this is not a one-unit bump. The optimal threshold departs from monotonicity by more than one unit or over most of the I₂ range, so this operator rewrites a large part of the policy. Its loss measures a structural mismatch rather than the cost of smoothing a local ridge.
  Cost: from (30, 2) at τ = T the expected cost rises from 208.9444 to 209.0550, a loss of 0.1107 (0.053%). The worst state at τ = T is (6, 3) with a loss of 0.3926 (0.074%); over all τ the largest loss is 0.3926 at τ = 4.812.
- remove M⁺: changed 996 cells in 331 periods. Monotonicity (b̄₁ non-increasing in I₂) is established; τ-direction (not enforced) has 5 violations afterwards.
  Reach: in the worst period b̄₁ was moved at 4 I₂ levels, by up to 1 units.
  The violation is a one-unit ridge, so this is a local smoothing of the optimal policy.
  Cost: from (30, 2) at τ = T the expected cost rises from 208.9444 to 209.6794, a loss of 0.7351 (0.352%). The worst state at τ = T is (4, 3) with a loss of 2.2135 (0.390%); over all τ the largest loss is 2.2135 at τ = 4.688.
- Comparison: fill M⁻ is cheaper at (30, 2) (0.1107 vs 0.7351). The two operators bracket the optimal table, so the cheaper one is the better monotone approximation of the two for this instance.
