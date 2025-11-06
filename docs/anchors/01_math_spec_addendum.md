What’s the real gap?
	•	Your current FREE transports cover: identity, D4/mirror-concat, v-double, tiling, and 3×7→3×3.
	•	Missing: selector-driven block substitution (SBS): the output is a grid of fixed-size blocks; which block template goes at (i,j) depends only on the input cell X_{ij} via a finite selector \sigma, while the template’s internal pattern is the Π-type mosaic taken from training outputs—not quotas from inputs.

Why ours missed it
	•	Our tile repeats the same type mosaic everywhere (σ ≡ constant).
	•	SBS needs σ(x) to choose one of M templates; tile ignores X, so on tasks like 007bbfb7 it “kron’s” the same template and fails.

Can we add SBS without breaking A0–A2?

Yes. It’s FREE transport iff all trainings prove the same finite transduction:

Proof obligation (per training pair X\!\to Y):
Find integers (s_h,s_w), a selector \sigma:\mathcal C\to\{1,\dots,M\}, and type-templates \{B^{(m)}\in\mathbb N^{s_h\times s_w}\}{m=1..M} (on types, not colors) s.t.
t_Y\!\big[i s_h:(i{+}1)s_h,\ j s_w:(j{+}1)s_w\big] \;=\; B^{(\sigma(X{ij}))}\quad\forall\,i,j,
where t_Y is the Π-type mosaic of Y.
	•	Verification is matching types only (FREE).
	•	With multiple trainings, intersect the proven (\sigma,\{B^{(m)}\}); if empty, reject SBS and fall back to the next simpler FREE map (keeps A0).

How to transport (test time)
	1.	Compute block grid on the test: same (s_h,s_w).
	2.	For each block (i,j), read x=X^\*_{ij} and place the type template B^{(\sigma(x))} at that block.
	3.	Disjointify type ids per block (so fills don’t cross between blocks).
	4.	Quotas remain the same (still counted from a training Y_0 per type); we didn’t mint any from X.

Why this closes the class (and stays simple)
	•	SBS strictly subsumes tiling (σ constant), blow-ups, and “mask stamping.”
	•	It preserves the orthogonality: Π + FREE transport (including SBS) on types, then paid quotas from outputs, then rank fill (meet).
	•	Complexity: linear in pixels; still minutes for all 1000.

Minimal FREE morphism library (credible “100% ARC” envelope)
	1.	Identity, D4 isometries, integer translations.
	2.	H/V concat; banded vertical/horizontal doubles.
	3.	Integer blow-ups (tile) and SBS (this fix).
	4.	Component-wise transports (copy/move a connected component’s types by FREE pose).
	5.	Stripe/column/row projections (3×7→3×3 family + row/col analogs).
	6.	Type-channel permutations (permute types only; colors stay quota-driven).
	7.	Disjoint relabeling of replicated types.

All are verifiable from one pair and intersectable across pairs; none reads quotas from X.

Receipts to prove it
	•	Π stencil + codebook hash.
	•	Verified FREE map kind (identity/…/SBS). For SBS: (s_h,s_w), σ table, and template type-hashes; intersection proof across trainings.
	•	Type partition sizes before/after transport; quotas K_{S,\cdot} (from a training Y).
	•	Output SHA256; optional determinacy UNSAT witness.
	•	Idempotence check (re-Π+fill on \(Y^\\) gives \(Y^\\)).
