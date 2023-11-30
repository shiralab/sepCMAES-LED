from abc import abstractmethod
from typing import *
from pathlib import Path
import traceback
import os
import shutil
import time
import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from objectives.objectivefactory import ContinuousObjectiveFunctionFactory
from optimizers._optimizer import _Optimizer
from optimizers.optimfactory import OptimizerFactory


def single_experiment(
    storagedir: Path,
    run_id: int,
    method: str,
    outdir: str,
    obj_name: str,
    dim: int,
    eff_dim: int,
    lam: int,
    n_iters_coef: float,
    interval: int,
    terminate_condition: int,
    kwargs: Dict[str, Any],
):
    outdir = outdir.replace("{:lam}", str(lam))
    savedir = storagedir / f"{obj_name}-{dim}-{eff_dim}" / outdir / str(run_id)
    os.makedirs(savedir, exist_ok=True)

    seed = run_id + 43
    best = np.inf
    objective_function = ContinuousObjectiveFunctionFactory.get(
        name=obj_name,
        dimensionality=dim,
        effective_dimensionality=eff_dim,
        terminate_condition=terminate_condition,
        seed=seed,
    )
    optimizer = OptimizerFactory.get(method=method, obj_name=obj_name, dim=dim, seed=seed, lam=lam)
    for k, v in kwargs.items():
        try:
            setattr(optimizer, k, v)
        except AttributeError as e:
            print(k)
            raise e

    is_success = False
    log = {"iter": [], "fcall": [], "feval": []}
    n_iters = dim * n_iters_coef // optimizer.pop_size + 1 if n_iters_coef > 0 else int(dim * 1e4 // optimizer.pop_size + 1)
    # t_imp = 0
    init_sigma = optimizer.sigma
    stack_thr = 10 + int(30 * objective_function.dimensionality / optimizer.pop_size)
    best_in_pop_hist_tol = np.ones(stack_thr) * np.inf
    hist_stag_min = int(120 + 30 * objective_function.dimensionality / optimizer.pop_size)
    hist_stag_max = 20000
    best_in_pop_hist_stag = np.array([])
    median_in_pop_hist_stag = np.array([])
    func_info = f"{objective_function.__class__.__name__} ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"

    try:
        for iter_i in range(n_iters):
            solutions = list()
            value_list = list()
            for _ in range(optimizer.pop_size):
                x = optimizer.ask()
                value = objective_function(x)
                value_list.append(value)
                try:
                    sample_idx = optimizer.sample_idx
                except:
                    sample_idx = -1

                if best > value:
                    best = value
                    print(
                        "\r",
                        f"#({optimizer.generations}, {optimizer.num_of_feval}) {np.round(best, 10)}, {func_info}",
                        end="",
                    )
                    if objective_function.is_optimized(x=x):
                        _ColorPrint.green(
                            f"\n[Optimized] #({optimizer.generations}, {optimizer.num_of_feval}) {np.round(best, 10)}, {func_info}"
                        )
                        is_success = True
                        break
                solutions.append((x, value, sample_idx))
            else:
                optimizer.tell(solutions=solutions)
                if iter_i % interval == 0:
                    log["iter"].append(iter_i)
                    log["fcall"].append(optimizer.num_of_feval)
                    log["feval"].append(best)
                
                # TolHistFun
                best_in_pop_hist_tol[iter_i % stack_thr] = np.min(value_list)
                if np.max(best_in_pop_hist_tol) - np.min(best_in_pop_hist_tol) < 1e-12:
                    _ColorPrint.yellow(
                        f"\n[Stacked (TolHistFun)] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, {func_info}"
                    )
                    break

                # Stagnation
                len_hist_stag = int( np.min( (np.max((0.2 * iter_i, hist_stag_min)), hist_stag_max) ) ) 
                best_in_pop_hist_stag = np.append(best_in_pop_hist_stag, np.min(value_list))
                median_in_pop_hist_stag = np.append(best_in_pop_hist_stag, np.median(value_list))
                if len(best_in_pop_hist_stag) > len_hist_stag:
                    best_in_pop_hist_stag = np.delete(best_in_pop_hist_stag, 0)
                if len(median_in_pop_hist_stag) > len_hist_stag:
                    median_in_pop_hist_stag = np.delete(median_in_pop_hist_stag, 0)
                
                if iter_i > hist_stag_min:
                    old_hist_median_best = np.median(best_in_pop_hist_stag[:int(0.3 * len_hist_stag)])
                    new_hist_median_best = np.median(best_in_pop_hist_stag[-int(0.3 * len_hist_stag):])
                    old_hist_median_median = np.median(median_in_pop_hist_stag[:int(0.3 * len_hist_stag)])
                    new_hist_median_median = np.median(median_in_pop_hist_stag[-int(0.3 * len_hist_stag):])
                    if old_hist_median_best <= new_hist_median_best and old_hist_median_median <= new_hist_median_median:
                        _ColorPrint.yellow(
                            f"\n[Stacked (Stagnation)] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, {func_info}"
                        )
                        break

                # TolX
                if optimizer.sigma * np.max((np.sqrt(np.max(optimizer.diagonal_C)), np.max(np.abs(optimizer.p_c)))) < 1e-12 * init_sigma:
                    _ColorPrint.yellow(
                        f"\n[Stacked (TolX)] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, {func_info}"
                    )
                    break
                # # ConditionCov
                # if np.max(optimizer.diagonal_C) / np.min(optimizer.diagonal_C) > 1e20: # 1e14 is too large for Ellipsoid(1024,2)
                #     _ColorPrint.yellow(
                #         f"\n[Stacked (ConditionCov)] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, {func_info}"
                #     )
                #     break
                continue
            break
        else:
            _ColorPrint.blue(
                f"\n[Not Converged] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, {func_info}"
            )
    except Exception as e:
        _ColorPrint.red(
            f"\n[Error] #({optimizer.generations}, {optimizer.num_of_feval}) {e}, {func_info}"
            + f"{traceback.format_exc()}"
        )
    pd.DataFrame(log).to_csv(savedir / "log.csv", index=False)
    (savedir / "success.txt").write_text(f"{is_success}\n")


class Experiment:
    _storage_dir: str
    _image_dir: str
    _obj_name: str
    _method: str
    _dim: int
    _eff_dim: int
    _history: "_History"
    _log_attrs: List[str]

    def __init__(self, file_path: str):
        self._storage_dir = os.path.join(os.path.abspath(os.path.dirname(file_path)), "storage")
        self._image_dir = os.path.join(os.path.abspath(os.path.dirname(file_path)), "images")

    def run(
        self,
        outdir: str,
        obj_name: str,
        method: str,
        dim: int,
        eff_dim: int,
        n_iter: int,
        n_run: int = 1,
        interval: int = 1,
        log_attrs: List[str] = [],
        lam: Optional[int] = None,
        terminate_condition: Optional[float] = 1e-7,
    ):
        """Run experiments and save the log to the directory.
        Created directory's name is '/[obj_name]-[dim]-[eff_dim]/[method]/[experiment No.]'.
        For example, '/Sphere-10-2/CMA-ES/1'.

        Args:
            outdir (str): output directory name
            obj_name (str): ex) Sphere, Ellipsoid
            method (str): ex) CMA-ES, ASNG
            dim (int): the num of dimensions
            eff_dim (int): the num of effective dimensions
            n_iter (int): the num of iterations. If it's reached, the experiment is terminated.
            n_run (int): the num of experiment to run.
            interval (int): the interval of iterations to log the results.
        """
        _ColorPrint.reverse(f"{obj_name} ({dim}, {eff_dim}), {method}")

        self._outdir = outdir.replace("{:lam}", str(lam))
        self._obj_name = obj_name
        self._method = method
        self._dim = dim
        self._eff_dim = eff_dim
        self._lam = lam
        self._terminate_condition = terminate_condition
        dir_name = self._make_save_dir()
        self._history = _History(
            columns=["func_eval", "n_func_eval", "x_opt_diff"] + log_attrs, base_dir_name=dir_name
        )
        self._log_attrs = log_attrs

        for run_i in range(n_run):
            self._make_folder(dir_name=os.path.join(dir_name, str(run_i + 1)), force=True)
            self._history.start_execution()

            best, t_imp = np.inf, 0
            objective_function = ContinuousObjectiveFunctionFactory.get(
                name=self._obj_name,
                dimensionality=self._dim,
                effective_dimensionality=self._eff_dim,
                terminate_condition=self._terminate_condition,
                seed=run_i + 6,
            )
            optimizer = self._configure_optimizer(seed=run_i + 5)

            stack_thr = 10 + int(30 * objective_function.dimensionality / optimizer.pop_size)
            best_in_pop_hist = np.ones(stack_thr) * np.inf
            func_info = f"{objective_function.__class__.__name__} ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"

            try:
                for i in range(n_iter):
                    t_imp += 1
                    solutions = list()
                    for _ in range(optimizer.pop_size):
                        x = optimizer.ask()
                        value = objective_function(x)
                        try:
                            sample_idx = optimizer.sample_idx
                        except:
                            sample_idx = -1

                        if best_in_pop > value:
                            best_in_pop = value

                        if best > value:
                            t_imp = 0
                            best = value
                            print(
                                "\r",
                                f"#({optimizer.generations}, {optimizer.num_of_feval}) {best}, ({objective_function.dimensionality}, {objective_function.effective_dimensionality})",
                                end="",
                            )
                            if objective_function.is_optimized(x=x):
                                _ColorPrint.green(
                                    f"\n[Optimized] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"
                                )
                                self._history.function_optimized()
                                break
                        solutions.append((x, value, sample_idx))
                    else:
                        optimizer.tell(solutions=solutions)
                        optimizer = self._additional_process(optimizer, objective_function)
                        if i % interval == 0:
                            self._save_history(
                                optimizer=optimizer,
                                best=best,
                                objective_function=objective_function,
                            )
                        if t_imp > self._stack_lim(optimizer=optimizer):
                            _ColorPrint.yellow(
                                f"\n[Stacked] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"
                            )
                            break
                        # if optimizer.sigma * np.min(optimizer.diagonal_C) < 1e-100:
                        #     _ColorPrint.yellow(
                        #         f"\n[Stacked] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"
                        #     )
                        #     break
                        best_in_pop_hist[i % stack_thr] = best_in_pop
                        if np.max(best_in_pop_hist) - np.min(best_in_pop_hist) < 1e-12:
                            _ColorPrint.yellow(
                                f"\n[Stacked] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, {func_info}"
                            )
                            break
                        continue
                    break
                else:
                    _ColorPrint.blue(
                        f"\n[Not Converged] #({optimizer.generations}, {optimizer.num_of_feval}) {best}, ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"
                    )
            except Exception as e:
                _ColorPrint.red(
                    f"\n[Error] #({optimizer.generations}, {optimizer.num_of_feval}) {e}, ({objective_function.dimensionality}, {objective_function.effective_dimensionality})"
                    + f"{traceback.format_exc()}"
                )
                # raise e
                continue
            finally:
                self._save_history(
                    optimizer=optimizer, best=best, objective_function=objective_function
                )
                self._history.terminated(folder_name=str(run_i + 1))
        self._history.end()
        print()

    def _configure_optimizer(self, seed: int = 100) -> _Optimizer:
        optimizer = OptimizerFactory.get(
            method=self._method, obj_name=self._obj_name, dim=self._dim, seed=seed, lam=self._lam
        )
        return optimizer

    def _additional_process(self, optimizer, objective_function):
        return optimizer

    def _stack_lim(self, optimizer: _Optimizer):
        return 10 + int(30 * optimizer.dimensionality / optimizer.pop_size)

    def _save_history(self, optimizer: _Optimizer, best: float, objective_function):
        self._history.save(col="func_eval", value=best)
        self._history.save(col="n_func_eval", value=optimizer.num_of_feval)
        self._history.save(
            col="x_opt_diff",
            value=optimizer.mean[: objective_function.x_opt.shape[0]] - objective_function.x_opt,
        )
        for attr in self._log_attrs:
            self._history.save(col=attr, value=getattr(optimizer, attr))

    def _make_save_dir(self) -> str:
        dir_name = os.path.join(self._storage_dir, f"{self._obj_name}-{self._dim}-{self._eff_dim}")
        self._make_folder(dir_name=dir_name, force=False)
        dir_name = os.path.join(dir_name, self._outdir)
        self._make_folder(dir_name=dir_name, force=True)
        _ColorPrint.white(f"-> {dir_name}")
        return dir_name

    def _make_folder(self, dir_name: str, force: bool):
        if os.path.isdir(dir_name):
            if force:
                shutil.rmtree(dir_name)
            else:
                return
        os.mkdir(dir_name)


class _ColorPrint:
    BLACK = "\033[30m"  # text color (black)
    RED = "\033[31m"  # text color (red)
    GREEN = "\033[32m"  # text color (green)
    YELLOW = "\033[33m"  # text color (yellow)
    BLUE = "\033[34m"  # text color (blue)
    MAGENTA = "\033[35m"  # text color (magenta)
    CYAN = "\033[36m"  # text color (cyan)
    WHITE = "\033[37m"  # text color (white)
    COLOR_DEFAULT = "\033[39m"  # reset text color as default
    BOLD = "\033[1m"  # bold font
    UNDERLINE = "\033[4m"  # underline
    INVISIBLE = "\033[08m"  # invisible font
    REVERSE = "\033[07m"  # switch the colors of text and background
    BG_BLACK = "\033[40m"  # background (black)
    BG_RED = "\033[41m"  # background (red)
    BG_GREEN = "\033[42m"  # background (green)
    BG_YELLOW = "\033[43m"  # background (yellow)
    BG_BLUE = "\033[44m"  # background (blue)
    BG_MAGENTA = "\033[45m"  # background (magenta)
    BG_CYAN = "\033[46m"  # background (cyan)
    BG_WHITE = "\033[47m"  # background (white)
    BG_DEFAULT = "\033[49m"  # reset background color as default
    RESET = "\033[0m"  # reset all

    @classmethod
    def red(cls, out: str):
        print(cls.RED + out + cls.RESET)

    @classmethod
    def green(cls, out: str):
        print(cls.GREEN + out + cls.RESET)

    @classmethod
    def blue(cls, out: str):
        print(cls.BLUE + out + cls.RESET)

    @classmethod
    def yellow(cls, out: str):
        print(cls.YELLOW + out + cls.RESET)

    @classmethod
    def white(cls, out: str):
        print(cls.WHITE + out + cls.RESET)

    @classmethod
    def reverse(cls, out: str):
        print(cls.REVERSE + out + cls.RESET)


class _History:
    start_at: int
    data: dict = {}
    is_function_optimized: bool = False
    success_dict = dict()
    base_dir_name: str

    def __init__(self, columns: list, base_dir_name: str):
        for col in columns:
            self.data[col] = list()
        self.base_dir_name = base_dir_name
        self.success_dict = {"trial": list(), "success": list()}

    def save(self, col: str, value):
        self.data[col].append(value)

    def start_execution(self):
        self.start_at = time.time()

    def terminated(self, folder_name: str):
        execution_time = time.time() - self.start_at
        self.success_dict["trial"].append(len(self.success_dict["trial"]) + 1)
        self.success_dict["success"].append(bool(self.is_function_optimized))
        for col in self.data:
            np.savetxt(
                os.path.join(self.base_dir_name, folder_name, "{}.csv".format(col)),
                np.array(self.data[col]),
                delimiter=",",
                fmt="%.18e",
            )
        self._clear()

    def function_optimized(self):
        self.is_function_optimized = True

    def end(self):
        success_df = pd.DataFrame(self.success_dict)
        success_df = success_df.set_index("trial")
        success_df.to_csv(os.path.join(self.base_dir_name, "success.csv"))

    def _clear(self):
        self.start_at = 0
        self.is_function_optimized = False
        for col in self.data:
            self.data[col] = list()
