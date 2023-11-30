import os, sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import json
import multiprocessing
from pathlib import Path
from concurrent import futures


from expt import single_experiment
from typing import *


def experiment(param: Dict[str, Any]):
    storagedir = Path(__file__).resolve().parent / "storage"
    single_experiment(storagedir=storagedir, **param)


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / sys.argv[1]
    config = json.load(open(config_path, "rb"))

    n_runs = config["n_runs"]
    n_workers = config["n_workers"] if config["n_workers"] > 0 else multiprocessing.cpu_count()
    method = config["method"]
    outdir = config["outdir"]

    params = list()
    for obj_name in config["obj_name"]:
        for dim, eff_dim, lam in zip(config["dim"], config["eff_dim"], config["lam"]):
            for run_id in range(n_runs):
                param = {
                    "run_id": run_id + 1,
                    "method": method,
                    "outdir": outdir,
                    "obj_name": obj_name,
                    "dim": dim,
                    "eff_dim": eff_dim,
                    "lam": lam,
                    "n_iters_coef": int(config["n_iters_coef"]),
                    "interval": config["interval"],
                    "terminate_condition": config["terminate_condition"],
                    "kwargs": {},
                }
                if "SepCMAESLED" == method:
                    param["kwargs"] = {
                        "gain_power_min": config["gain_power_min"],
                        "gain_power_max": config["gain_power_max"],
                        "beta_hat": config["beta_hat"],
                    }
                if "SepCMAESModifiedTPALED" == method:
                    param["kwargs"] = {
                        "gain_power_min": config["gain_power_min"],
                        "gain_power_max": config["gain_power_max"],
                        "beta_hat": config["beta_hat"],
                    }
                params.append(param)

    print(f"Max Workers: {n_workers}")
    pool = futures.ProcessPoolExecutor(max_workers=n_workers)
    result = list(pool.map(experiment, params))