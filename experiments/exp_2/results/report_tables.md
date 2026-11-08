## Main table

**original**

| arm | mean spellF1 | min | macroF1 | mean / max abs count err | Δ vs exp1_ref [95% CI] | wins | count gate |
|---|---|---|---|---|---|---|---|
| `exp1_ref` | 0.656 | 0.616 | 0.918 | 0.212 / 0.342 | – | – | FAIL |
| `exp2_aug` | 0.657 | 0.596 | 0.917 | 0.200 / 0.354 | +0.001 [-0.008, +0.010] | 3/7 | FAIL |
| `exp2_aug_const` | 0.657 | 0.596 | 0.917 | 0.200 / 0.354 | +0.001 [-0.008, +0.010] | 3/7 | FAIL |
| `exp2_aug_hsmm` | 0.656 | 0.599 | 0.917 | 0.190 / 0.337 | +0.001 [-0.009, +0.010] | 4/7 | FAIL |
| `exp2_aug_s1` | 0.651 | 0.617 | 0.918 | 0.192 / 0.344 | -0.005 [-0.016, +0.006] | 3/7 | FAIL |
| `exp2_aug_s2` | 0.656 | 0.618 | 0.917 | 0.220 / 0.335 | +0.001 [-0.008, +0.011] | 3/7 | FAIL |
| `exp2_aug_ens3` | 0.666 | 0.621 | 0.922 | 0.204 / 0.360 | +0.011 [+0.004, +0.019] | 6/7 | FAIL |
| `exp1_ref_argmax` | 0.624 | 0.557 | 0.918 | 0.089 / 0.182 | -0.032 [-0.043, -0.022] | 0/7 | pass |
| `exp2_aug_argmax` | 0.628 | 0.552 | 0.917 | 0.082 / 0.209 | -0.027 [-0.043, -0.014] | 0/7 | pass |
| `ryu_final` | 0.487 | 0.404 | 0.879 | 0.457 / 0.528 | -0.168 [-0.202, -0.132] | 0/7 | FAIL |
| `rasya_c08` | 0.325 | 0.308 | 0.591 | 0.411 / 0.489 | -0.331 [-0.351, -0.309] | 0/7 | FAIL |
| `rasya_c06` | 0.322 | 0.301 | 0.634 | 0.357 / 0.464 | -0.333 [-0.349, -0.315] | 0/7 | FAIL |
| `baseline` | 0.186 | 0.167 | 0.458 | 0.179 / 0.354 | -0.469 [-0.487, -0.450] | 0/7 | FAIL |

**target_like**

| arm | mean spellF1 | min | macroF1 | mean / max abs count err | Δ vs exp1_ref [95% CI] | wins | count gate |
|---|---|---|---|---|---|---|---|
| `exp1_ref` | 0.648 | 0.598 | 0.913 | 0.204 / 0.342 | – | – | FAIL |
| `exp2_aug` | 0.657 | 0.604 | 0.917 | 0.199 / 0.354 | +0.009 [+0.001, +0.018] | 5/7 | FAIL |
| `exp2_aug_const` | 0.657 | 0.604 | 0.917 | 0.199 / 0.354 | +0.009 [+0.001, +0.018] | 5/7 | FAIL |
| `exp2_aug_hsmm` | 0.654 | 0.592 | 0.917 | 0.185 / 0.341 | +0.006 [-0.003, +0.016] | 5/7 | FAIL |
| `exp2_aug_s1` | 0.649 | 0.610 | 0.917 | 0.192 / 0.352 | +0.001 [-0.008, +0.010] | 4/7 | FAIL |
| `exp2_aug_s2` | 0.656 | 0.607 | 0.917 | 0.223 / 0.337 | +0.008 [-0.002, +0.019] | 5/7 | FAIL |
| `exp2_aug_ens3` | 0.664 | 0.615 | 0.921 | 0.202 / 0.366 | +0.016 [+0.010, +0.023] | 7/7 | FAIL |
| `exp1_ref_argmax` | 0.616 | 0.542 | 0.912 | 0.086 / 0.170 | -0.032 [-0.043, -0.022] | 0/7 | pass |
| `exp2_aug_argmax` | 0.627 | 0.550 | 0.917 | 0.085 / 0.223 | -0.021 [-0.036, -0.007] | 2/7 | pass |
| `ryu_final` | 0.479 | 0.401 | 0.870 | 0.446 / 0.523 | -0.169 [-0.203, -0.133] | 0/7 | FAIL |
| `rasya_c08` | 0.266 | 0.232 | 0.374 | 0.524 / 0.622 | -0.382 [-0.403, -0.353] | 0/7 | FAIL |
| `rasya_c06` | 0.260 | 0.226 | 0.403 | 0.499 / 0.607 | -0.388 [-0.406, -0.364] | 0/7 | FAIL |
| `baseline` | 0.204 | 0.182 | 0.614 | 0.234 / 0.431 | -0.443 [-0.466, -0.421] | 0/7 | FAIL |


## Rules

| rule | Δ [95% CI] / statistic | result |
|---|---|---|
| H1a_transfer_target_like (CI lower > 0) | +0.009 [+0.001, +0.018], wins 5/7 | **passed** |
| H1b_in_domain_original (mean delta >= -0.02) | +0.001 [-0.008, +0.010], wins 3/7 | **passed** |
| H2_count_gate_target_like (all 7 matches within +-30%) | max abs count error 0.341 | **failed** |
| H2_non_inferiority_target_like (CI lower > -0.01 vs constant decoder) | -0.002 [-0.006, +0.001], wins 4/7 | **passed** |
| H3_ensemble_target_like (|mean delta| > seed spread and CI excludes 0) | +0.007 [+0.001, +0.014], wins 5/7, seed spread 0.007 | **passed** |

Supported: {'H1': True, 'H2': False, 'H3': True}; adopt exp_2 candidate: True

## Per match

**original** (spellF1; count error in brackets)

| match | `exp1_ref` | `exp2_aug` | `exp2_aug_hsmm` | `exp2_aug_ens3` | `ryu_final` |
|---|---|---|---|---|---|
| J03WMX | 0.617 (-0.34) | 0.630 (-0.35) | 0.631 (-0.34) | 0.623 (-0.36) | 0.449 (-0.53) |
| J03WN1 | 0.644 (-0.29) | 0.641 (-0.27) | 0.642 (-0.25) | 0.655 (-0.30) | 0.404 (-0.50) |
| J03WOH | 0.675 (-0.21) | 0.675 (-0.22) | 0.675 (-0.22) | 0.672 (-0.24) | 0.514 (-0.46) |
| J03WOY | 0.666 (-0.18) | 0.687 (-0.15) | 0.688 (-0.13) | 0.699 (-0.16) | 0.449 (-0.48) |
| J03WPY | 0.683 (-0.19) | 0.688 (-0.17) | 0.686 (-0.16) | 0.699 (-0.19) | 0.509 (-0.45) |
| J03WQQ | 0.688 (-0.04) | 0.680 (-0.06) | 0.672 (-0.08) | 0.694 (-0.05) | 0.542 (-0.39) |
| J03WR9 | 0.616 (-0.23) | 0.596 (-0.16) | 0.599 (-0.15) | 0.621 (-0.14) | 0.544 (-0.39) |

**target_like** (spellF1; count error in brackets)

| match | `exp1_ref` | `exp2_aug` | `exp2_aug_hsmm` | `exp2_aug_ens3` | `ryu_final` |
|---|---|---|---|---|---|
| J03WMX | 0.604 (-0.34) | 0.621 (-0.35) | 0.622 (-0.34) | 0.615 (-0.37) | 0.429 (-0.52) |
| J03WN1 | 0.639 (-0.27) | 0.639 (-0.24) | 0.640 (-0.25) | 0.659 (-0.29) | 0.401 (-0.50) |
| J03WOH | 0.667 (-0.20) | 0.668 (-0.21) | 0.670 (-0.18) | 0.670 (-0.23) | 0.510 (-0.46) |
| J03WOY | 0.662 (-0.16) | 0.696 (-0.14) | 0.691 (-0.12) | 0.695 (-0.16) | 0.450 (-0.47) |
| J03WPY | 0.677 (-0.20) | 0.684 (-0.18) | 0.689 (-0.16) | 0.691 (-0.18) | 0.503 (-0.42) |
| J03WQQ | 0.688 (-0.05) | 0.685 (-0.07) | 0.677 (-0.09) | 0.698 (-0.06) | 0.530 (-0.40) |
| J03WR9 | 0.598 (-0.21) | 0.604 (-0.19) | 0.592 (-0.15) | 0.621 (-0.13) | 0.528 (-0.36) |


## View shift (target_like - original)

* `exp1_ref`: -0.008 [-0.012, -0.004]
* `exp2_aug`: -0.000 [-0.005, +0.005]
* `exp2_aug_const`: -0.000 [-0.005, +0.005]
* `exp2_aug_hsmm`: -0.002 [-0.005, +0.002]
* `exp2_aug_s1`: -0.002 [-0.005, +0.002]
* `exp2_aug_s2`: -0.001 [-0.007, +0.003]
* `exp2_aug_ens3`: -0.002 [-0.005, +0.001]
* `exp1_ref_argmax`: -0.008 [-0.011, -0.005]
* `exp2_aug_argmax`: -0.002 [-0.005, +0.001]
* `ryu_final`: -0.009 [-0.014, -0.004]
* `rasya_c08`: -0.059 [-0.073, -0.045]
* `rasya_c06`: -0.062 [-0.072, -0.053]
* `baseline`: +0.018 [+0.014, +0.024]

## Gates

| view        | variant         | gain_at_least_005   | ci_lower_above_zero   | macro_regression_at_most_001   | all_match_counts_within_30pct   | pass   | worst       |
|:------------|:----------------|:--------------------|:----------------------|:-------------------------------|:--------------------------------|:-------|:------------|
| original    | exp1_ref        | True                | True                  | True                           | False                           | False  | J03WMX 0.34 |
| original    | exp2_aug        | True                | True                  | True                           | False                           | False  | J03WMX 0.35 |
| original    | exp2_aug_const  | True                | True                  | True                           | False                           | False  | J03WMX 0.35 |
| original    | exp2_aug_hsmm   | True                | True                  | True                           | False                           | False  | J03WMX 0.34 |
| original    | exp2_aug_s1     | True                | True                  | True                           | False                           | False  | J03WMX 0.34 |
| original    | exp2_aug_s2     | True                | True                  | True                           | False                           | False  | J03WMX 0.33 |
| original    | exp2_aug_ens3   | True                | True                  | True                           | False                           | False  | J03WMX 0.36 |
| original    | exp1_ref_argmax | True                | True                  | True                           | True                            | True   | J03WMX 0.18 |
| original    | exp2_aug_argmax | True                | True                  | True                           | True                            | True   | J03WMX 0.21 |
| original    | ryu_final       | True                | True                  | True                           | False                           | False  | J03WMX 0.53 |
| original    | rasya_c08       | True                | True                  | True                           | False                           | False  | J03WMX 0.49 |
| original    | rasya_c06       | True                | True                  | True                           | False                           | False  | J03WMX 0.46 |
| target_like | exp1_ref        | True                | True                  | True                           | False                           | False  | J03WMX 0.34 |
| target_like | exp2_aug        | True                | True                  | True                           | False                           | False  | J03WMX 0.35 |
| target_like | exp2_aug_const  | True                | True                  | True                           | False                           | False  | J03WMX 0.35 |
| target_like | exp2_aug_hsmm   | True                | True                  | True                           | False                           | False  | J03WMX 0.34 |
| target_like | exp2_aug_s1     | True                | True                  | True                           | False                           | False  | J03WMX 0.35 |
| target_like | exp2_aug_s2     | True                | True                  | True                           | False                           | False  | J03WMX 0.34 |
| target_like | exp2_aug_ens3   | True                | True                  | True                           | False                           | False  | J03WMX 0.37 |
| target_like | exp1_ref_argmax | True                | True                  | True                           | True                            | True   | J03WMX 0.17 |
| target_like | exp2_aug_argmax | True                | True                  | True                           | True                            | True   | J03WMX 0.22 |
| target_like | ryu_final       | True                | True                  | True                           | False                           | False  | J03WMX 0.52 |
| target_like | rasya_c08       | True                | True                  | False                          | False                           | False  | J03WMX 0.62 |
| target_like | rasya_c06       | True                | True                  | False                          | False                           | False  | J03WMX 0.61 |
