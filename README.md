# sep-CMA-ES-LED
This is implementation of sep-CMA-ES-LED [[DOI](https://ieeexplore.ieee.org/document/10022244)].

## Sample Code
- Code: `sample.ipynb`
- Comparison with sep-CMA-ES and sep-CMA-ES-LED

## Run Experiments
```
cd experiments
python multirun.py configs/[config file name]
```
- The result is saved in `experiments/storage`
- config file names (see `experiments/configs`)
    - sep-CMAES (CSA) : `multi-config.json`
    - sep-CMAES-LED (CSA) : `multi-config-led.json`
    - sep-CMAES (TPA) : `multi-config-mod-tpa.json`
    - sep-CMAES-LED (TPA) : `multi-config-mod-tpa-led.json`

## Config File
```
{
    "n_runs": 1,                        # num. of trials
    "n_workers": 0,                     # num. of thread for multiprocessing（"0" means the maximum)
    "obj_name": [
        "Sphere", "Ellipsoid",          # functions
    ],
    "outdir": "SepCMAES",               # log file name (saved in storage/[file name])
    "method": "SepCMAES",               # optimization method
    "dim": [
        4, 8                            # total dimensions（the size of "dim", "eff_dim", "lam" should be same）
    ],
    "eff_dim": [
        4, 4                            # effective dimensions（the size of "dim", "eff_dim", "lam" should be same）
    ],
    "lam": [
        0, 0                            # sample size（"0" means the default value）
    ],
    "n_iters": 0,                       # maximum num. of iterations（"0" means "dim" x 10^4)
    "interval": 1,                      # interval for log (iteration)
    "beta_hat": 0.01,                   # accumulation rate
    "gain_power_min": -1,               # g_min
    "gain_power_max": 5,                # g_max
    "terminate_condition": 1e-8         # terminate condition
}
```

## States for Terminate Conditions
- Optimized: the optimizer satisfied the terminate_condition 
- Stucked: the best evaluation value was not improved for some iterations
- Not Converged: the optimizer reached maximum iteration 
- Error: the optimizer occurred some error

## Environment
```
Jinja2==3.1.1
json5==0.9.6
numpy==1.22.3
pandas==1.4.1
scipy==1.8.0
```

## Reference
T. Yamaguchi, K. Uchida and S. Shirakawa, "Improvement of sep-CMA-ES for Optimization of High-Dimensional Functions with Low Effective Dimensionality," 2022 IEEE Symposium Series on Computational Intelligence (SSCI), Singapore, Singapore, 2022, pp. 1659-1668, doi: [10.1109/SSCI51031.2022.10022244](https://doi.org/10.1109/SSCI51031.2022.10022244)