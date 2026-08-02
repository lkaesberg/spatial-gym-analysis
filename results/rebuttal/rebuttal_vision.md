## Vision ablation across model families (Spatial-Gym)

Accuracy = "Correctly Solved" %. Δ = vision − text (negative = vision hurts).

| Model | Text Acc. (%) | Vision Acc. (%) | Δ (pp) |
|---|---:|---:|---:|
| Qwen3-VL 32B (ref.) | 10.2 | 2.8 | -7.4 |
| Gemma 3 27B | 4.0 | 1.6 | -2.4 |
| Gemma 4 31B | 13.8 | 6.2 | -7.6 |
| Magistral Small$^\dagger$ | 2.6 | 2.0 | -0.6 |

**Notes.**
- (ref.) Qwen3-VL 32B is the original single-family ablation, shown for reference.
- $^\dagger$ The Magistral vision run is **Magistral-Small-2509**, compared against the **Magistral-Small-2507** text baseline in our pool (version mismatch).

**Observation.** The text→vision accuracy drop is **not** specific to the Qwen family: every model loses accuracy under the visual setting, with the largest drop for the strongest text model (Gemma 4 31B).
