# Pointcloud registration
---

## Iterative closest point (ICP) algorithm results

Two pointclouds of walls, with high overlap
<img src="resultPhotos/wallsStart.png" width="800"/>

Blue pointcloud moved (green) to fit the red one
<img src="resultPhotos/wallsRes.png" width="800"/>

---

A pointcloud (red) and a translated, rotated, noisy copy of it (blue)
<img src="resultPhotos/fountainStart.png" width="800"/>

Blue pointcloud moved (green) to fit the red one
<img src="resultPhotos/fountainRes.png" width="800"/>


---

Results when applied to partially overlapping pointclouds. Mean squared error decreases, but the result is meaningless.
The green pointcloud moves "closer" but is not aligned.
<img src="resultPhotos/ICPbad.png" width="800"/>


## Trimmed Iterative closest point (TrICP) algorithm

Aligning based on only the top 60% closest pairs, the results are correct (MSE = 0.0043)
<img src="resultPhotos/trRes.png" width="800"/>

Starting position (MSE = 0.0606)
<img src="resultPhotos/trStart.png" width="800"/>
