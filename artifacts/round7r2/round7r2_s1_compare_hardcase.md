# Run Comparison

## mbpp_eval
| mode | acc | avg_cost | p50 | p95 | router_p50 | llm_p50 | test_p50 | calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| masrouter | 0.8250 | 0.00074717 | 17.905 | 52.848 | 0.011 | 17.831 | 0.034 | 280 |
| masrouter_cheap | 0.7875 | 0.00053751 | 11.792 | 37.603 | 0.011 | 11.749 | 0.035 | 200 |
| masrouter_balanced | 0.8250 | 0.00074717 | 17.905 | 52.848 | 0.011 | 17.831 | 0.034 | 280 |
| masrouter_premium | 0.8000 | 0.00111314 | 31.034 | 76.703 | 0.011 | 30.662 | 0.035 | 360 |
| wae_static_cheap | 0.8750 | 0.00027857 | 7.671 | 13.030 | 5.082 | 2.468 | 0.032 | 172 |
| wae_static_premium | 0.8250 | 0.00048709 | 11.473 | 44.452 | 6.606 | 2.747 | 0.029 | 356 |
| wae_dynamic_no_premium | 0.7750 | 0.00025689 | 7.471 | 18.007 | 4.558 | 2.094 | 0.035 | 160 |
| wae_dynamic_hardcase_gate | 0.7875 | 0.00031810 | 7.946 | 16.459 | 5.203 | 2.766 | 0.030 | 402 |
| wae_dynamic_control_forced_io | 0.7625 | 0.00024058 | 7.124 | 12.804 | 3.900 | 2.072 | 0.032 | 160 |
| wae_cheap_first_escalate | 0.8875 | 0.00040100 | 9.493 | 34.464 | 6.079 | 2.484 | 0.035 | 214 |
| wae_dynamic | 0.7875 | 0.00027017 | 7.828 | 14.239 | 5.062 | 2.234 | 0.030 | 160 |

- iso-cost method: `linear_interpolation` (comparable=`True`)
- target mode: `wae_dynamic_hardcase_gate`
- baseline envelope size: `5`
- delta acc (wae_dynamic_hardcase_gate - reference): `-0.0915`
- success(+0.03): `False`
- dominated on (cost, acc): `True` by `['wae_dynamic', 'wae_static_cheap']`
- dominated on (cost, acc, p50lat): `True` by `['wae_dynamic', 'wae_static_cheap']`
- final verdict (dominance-first): `False` (`dominated_by_baseline`)
- plot: `/workspace/wae_router_pilot/runs/round7r2_s1_compare_hardcase_cost_acc_mbpp_eval.png`
- budget best-under-budget:
  - B=0.00026000: best=`wae_dynamic_no_premium` acc=`0.775` feasible=2
  - B=0.00030000: best=`wae_static_cheap` acc=`0.875` feasible=4
  - B=0.00040000: best=`wae_static_cheap` acc=`0.875` feasible=5

## humaneval_eval
| mode | acc | avg_cost | p50 | p95 | router_p50 | llm_p50 | test_p50 | calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| masrouter | 0.7625 | 0.00153457 | 32.782 | 78.452 | 0.011 | 32.736 | 0.034 | 280 |
| masrouter_cheap | 0.8375 | 0.00092345 | 20.738 | 41.075 | 0.011 | 20.690 | 0.035 | 200 |
| masrouter_balanced | 0.7625 | 0.00153457 | 32.782 | 78.452 | 0.011 | 32.736 | 0.034 | 280 |
| masrouter_premium | 0.7250 | 0.00234080 | 55.770 | 114.040 | 0.011 | 55.724 | 0.033 | 360 |
| wae_static_cheap | 0.8375 | 0.00043644 | 9.464 | 21.874 | 5.933 | 2.719 | 0.035 | 175 |
| wae_static_premium | 0.8375 | 0.00159330 | 34.967 | 73.693 | 26.090 | 7.650 | 0.029 | 365 |
| wae_dynamic_no_premium | 0.8750 | 0.00038854 | 7.648 | 19.723 | 4.877 | 2.408 | 0.030 | 160 |
| wae_dynamic_hardcase_gate | 0.8625 | 0.00048139 | 8.245 | 28.340 | 5.521 | 2.369 | 0.034 | 372 |
| wae_dynamic_control_forced_io | 0.8875 | 0.00036943 | 7.587 | 17.572 | 4.719 | 2.409 | 0.030 | 160 |
| wae_cheap_first_escalate | 0.9250 | 0.00069899 | 10.015 | 50.825 | 6.727 | 2.668 | 0.035 | 216 |
| wae_dynamic | 0.8875 | 0.00037217 | 47.263 | 68.039 | 43.881 | 2.719 | 0.219 | 160 |

- iso-cost method: `linear_interpolation` (comparable=`True`)
- target mode: `wae_dynamic_hardcase_gate`
- baseline envelope size: `2`
- delta acc (wae_dynamic_hardcase_gate - reference): `-0.0377`
- success(+0.03): `False`
- dominated on (cost, acc): `True` by `['wae_dynamic', 'wae_dynamic_control_forced_io', 'wae_dynamic_no_premium']`
- dominated on (cost, acc, p50lat): `True` by `['wae_dynamic_control_forced_io', 'wae_dynamic_no_premium']`
- final verdict (dominance-first): `False` (`dominated_by_baseline`)
- plot: `/workspace/wae_router_pilot/runs/round7r2_s1_compare_hardcase_cost_acc_humaneval_eval.png`
- budget best-under-budget:
  - B=0.00026000: best=`None` acc=`None` feasible=0
  - B=0.00030000: best=`None` acc=`None` feasible=0
  - B=0.00040000: best=`wae_dynamic` acc=`0.8875` feasible=3
