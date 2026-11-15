# Low-rim cap independence — offline review

This report changes no simulation source, configuration, force limit, contact gate, or validator. It does not certify a completed insertion. The executable companion is `low_rim_cap_independence_review.py`; its measurements are in `low-rim-cap-independence-review.json`.

## Conclusion

The present `validate.py:containment` rejection of every SVD-projection crossing is stronger than necessary **for query points strictly below every actual rim vertex**. In that restricted region, the existing coherently oriented mean-center triangle fan gives a cap-independent winding number even when the cap overlaps itself or its two-dimensional rim projection crosses.

This is a statement about the winding integral of specified query points. Turning that number into a physical inside/outside certificate still requires the physical cloth body to represent an appropriate nonintersecting boundary. The theorem does not repair crossed cloth, a wrong topology, inverted face ordering, a teddy penetrating cloth, an incomplete release, or unstable final settling.

## Exact statement and proof

Let `S` be the oriented physical body triangle chain and `L = boundary(S)` its one mouth boundary. Set `h = min(rim[:,2])` and `H = {x : x.z >= h}`. Let `C1` and `C2` be finite oriented triangle chains supported entirely in `H`, both having boundary `-L`. For a point `q` with `q.z < h`, not on `S`,

`W(S + C1, q) = W(S + C2, q)`.

**Proof.** The difference `K = C1 - C2` has zero boundary. The half-space `H` is convex and excludes `q`. Coning `K` to any point in `H` constructs an oriented 3-chain `B` in `H` with boundary `K`; self-overlaps are counted with their signs. The solid-angle flux field `(x-q)/(4*pi*|x-q|^3)` is smooth and has divergence zero throughout `H`. The divergence theorem applied to this chain gives `W(K,q)=0`. Additivity of the surface integral yields the result. Equivalently, a closed chain can contract to a point within `H` without crossing `q`, so its degree at `q` is zero.

The arithmetic rim mean is in the convex hull of the rim vertices. Each triangle connecting two rim vertices to that mean is also in their convex hull. Every vertex, and therefore every point of every fan triangle, has `z >= h`. Thus the existing fan satisfies the required height property without assuming a planar, simple, convex, or star-shaped rim projection. Correct directed boundary cancellation remains essential. A curved cap with control points below `h` would not inherit this guarantee merely because its boundary is high.

If there is a genuine embedded closure `C*` within `H`, disjoint from the physical body away from their common boundary, and `S+C*` is the intended oriented boundary of the bag interior, then the existing fan has exactly the same winding number at every such low query as that physical closure. This is a sufficient condition for interpreting unit winding as physical containment. Cap independence alone does **not** prove existence of such an embedded closure or absence of physical body self-intersections. A topologically watertight, consistently indexed triangle soup can still be geometrically self-intersecting.

The solid-angle/winding formulation and the distinction between arbitrary oriented meshes and proper watertight boundaries are discussed by [Jacobson, Kavan, and Sorkine-Hornung, 2013](https://igl.ethz.ch/projects/winding-number/). The separating-half-space argument above is this review's derivation, not a claim quoted from that source.

## Conditions for any future, explicit low-query branch

1. Keep the exact original body labels, one mouth boundary, two sewn panels, connectedness/manifold topology, face orientation, and directed boundary cancellation checks. The fan rim set must equal the physical body's boundary set. No bottom or side panel is introduced.
2. Use the actual current state. Require every teddy vertex **and** its mass center below the actual lowest rim with an explicit positive margin. A 2 mm margin would match the present planned endpoint clearance; its actual attainment must be checked. For a linear tetrahedral mesh this height condition at all nodes also bounds the height of every interpolated point in every tetrahedron.
3. Check finite coordinates, nondegenerate relevant geometry, and separation from the physical cloth. Retain genuine 3D rim nonintersection/separation checks. A 2D crossing count and projected area may remain diagnostics; they do not establish physical 3D intersection or determine this theorem.
4. Preserve every existing native contact/CCD, force/torque, strain, table penetration, grasp, robot clearance, release, and final stable-state check. The theorem does not supply these properties. Evidence of an actual physical body self-intersection must still reject physical containment, regardless of a numerical winding value.
5. Retain raw winding values. For a coherently closed chain away from its triangles they should be close to integers. Reject ambiguous near-surface or nonfinite evaluations, anomalous integer residuals, and multiple winding such as `abs(w)=2`; the current bare `abs(w)>.5` predicate is not by itself a certificate of a simple physical interior. A deliberate numerical tolerance must be documented and checked against geometry scale and separation; these small synthetic tests do not establish a universal tolerance.
6. An optional second cap with an elevated apex in `H` should agree numerically on all low queries. This is a useful implementation/orientation sanity check, **not** independent proof of physical cloth nonintersection.
7. Keep all teddy vertices and COM inside, the original final cloth/teddy sampled penetration check, and sustained final-state checks. All vertices and COM inside a nonconvex bag do not mathematically prove every edge, face, and tetrahedron inside; the existing contact audit is discrete and must retain that limitation. A connected tetrahedron can span a concave exterior notch even when its vertices are inside. This report does not upgrade discrete sampling to continuous collision certification.

Such a branch should describe its result as **cap-independent low-region winding**, and report the positive height margin, actual winding values, raw projection diagnostics, and remaining physical verification scope. Calling a self-overlapping fan an embedded valid closure would overstate what was proved.

## Seven bounded synthetic tests

| Case | Observation |
|---|---|
| Ordinary open cup; three caps entirely above the rim floor | Interior/exterior queries below the rim agree to `2.22e-16`. |
| True 3D simple rim with one crossed SVD projection | Nonadjacent rim edges remain 0.1 m apart; three cap contributions agree to `1.39e-16`. This fixture tests cap independence, not a physical body. |
| Query above the lowest rim | Flat versus high cap changes winding from 0 to 1: the height restriction is necessary. |
| Cap apex lowered below the rim | A low query changes winding from 1 to 0: boundary height alone is insufficient for an arbitrary cap. |
| One cap triangle reversed | Index topology stays watertight, orientation consistency fails, and winding becomes `0.9103767`. |
| Two overlapping coherently oriented closed boxes | Watertightness and orientation flags both pass while overlap winding is 2; these flags alone do not prove embedding. The synthetic fixture has two components and is not asserted to satisfy the real bag's topology. |
| Low-plane gaps 2 mm, 1 micrometer, and 1 nanometer on a simple cup | The evaluated caps agree; this small fixture does not establish a production floating-point error bound. |

All seven checks passed. No GPU or physical simulation was run.

## Actual full14: theorem is not applicable to the whole bear

At the actual 54.085 s entry endpoint:

- Lowest rim: **1.007195058 m**.
- Highest teddy vertex: **1.222238284 m**.
- Whole-teddy height margin: **−0.215043226 m**. The bear is not wholly below the rim.
- The actual SVD rim projection has **2 crossings**.
- **67 of 1450** teddy vertices lie at least 2 mm below the lowest rim. On this limited subset, the mean fan and an elevated-apex fan differ by at most **4.42e-17** in winding contribution.
- Across all teddy vertices, the two cap contributions differ by as much as **1.00000000000005**, which directly demonstrates why the low-height condition cannot be omitted at this partial insertion state.

This review therefore supplies **no full14 containment pass** and no successful task result. A future actual post-insertion state must independently satisfy the height precondition and every original physical/release/stability requirement. The reviewed `validate.py` SHA-256 is `e077a79f02bef7779e79519f00283ec8da7fc8df53f9cc708386ec40567fad99`; the companion JSON also records the exact full14 input hashes.
