from __future__ import annotations
from typing import List
import os
import sys
import shutil
import time
import numpy as np
import base64
from jinja2 import Environment, FileSystemLoader

STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
    )
)


def make_folder(dir_name: str, force: bool):
    if os.path.isdir(dir_name):
        if force:
            shutil.rmtree(dir_name)
        else:
            return
    os.mkdir(dir_name)


def make_images_folder(dir_name: str, force=True):
    image_folder = os.path.join(IMAGES, dir_name)
    make_folder(dir_name=image_folder, force=force)


def make_experiment_folder(dir_name: str, force=True):
    expt_folder = os.path.join(STORAGE, dir_name)
    make_folder(dir_name=expt_folder, force=force)


def fetch_success_first_indexes(dir_name: str) -> tuple:
    data = np.loadtxt(
        os.path.join(os.path.join(STORAGE, dir_name), "success_list.csv"), dtype=str
    )[:-1]
    suc_index, suc_list = data[:, 0], data[:, 1]
    first_success_index, first_fail_index = None, None
    if np.sum(suc_list == "True") > 0:
        first_success_index = suc_index[suc_list == "True"][0]
    if np.sum(suc_list == "False") > 0:
        first_fail_index = suc_index[suc_list == "False"][0]
    return first_success_index, first_fail_index


def fetch_success_rate(dir_name: str):
    data = np.loadtxt(
        os.path.join(os.path.join(STORAGE, dir_name), "success_list.csv"), dtype=str
    )[:-1]
    suc_list = data[:, 1]
    return float(np.sum(suc_list == "True") / len(suc_list))


def fetch_success_indexes(dir_name: str):
    data = np.loadtxt(
        os.path.join(os.path.join(STORAGE, dir_name), "success_list.csv"), dtype=str
    )[:-1]
    suc_index, suc_list = data[:, 0], data[:, 1]
    return suc_index[suc_list == "True"]


def fetch_success_rate_(dir_name: str):
    data = np.loadtxt(os.path.join(dir_name, "success_list.csv"), dtype=str)[:-1]
    suc_list = data[:, 1]
    return float(np.sum(suc_list == "True") / len(suc_list))


def fetch_success_indexes_(dir_name: str):
    data = np.loadtxt(os.path.join(dir_name, "success_list.csv"), dtype=str)[:-1]
    suc_index, suc_list = data[:, 0], data[:, 1]
    return suc_index[suc_list == "True"]


def get_hyper_parameter(h_type: str, dim: int):
    if h_type == "a":
        return 1 / dim
    elif h_type == "b":
        return 1 / np.sqrt(dim)
    elif h_type == "c":
        return 1.0
    elif h_type == "d":
        return np.sqrt(dim)


def get_arguments(params: list, indexes: List[int]):
    """
    when params doesn't have element in indexes, return the list replaced with None
    :param params: list sys.args
    :param indexes: list
    :return:
    """
    return [params[idx] if idx <= len(params) - 1 else None for idx in indexes]


def arguments_to_str(param_nms: List[str], params: List[str]):
    """
    float value to string
    ex) beta=0.05, delta=0.01 → beta005_delta001
    :param param_nms:
    :param params:
    :return: string
    """
    return "_".join(
        [p_nm + p_vl.replace(".", "") for p_nm, p_vl in zip(param_nms, params)]
    )


def get_exponent_of_dimension(dim: int, exp_indexes: str):
    """
    get exponent of the number of dimensions
    :param dim: int
    :param exp_indexes: List[number]
    :return: List[number]
    """
    return [dim ** float(exp_idx) for exp_idx in exp_indexes]


def exponent_of_dimension_to_str(exp_indexes: List[str]):
    """
    ex) '0.5' → '$D^{0.5}$'
    :param exp_indexes:
    :return:
    """
    return ["$D^{" + exp_idx + "}$" for exp_idx in exp_indexes]


def build_env_with_img_decoder(dir_name: str) -> Environment:
    """
    :param dir_name: directory where template file is
    :return: Environment
    """
    env = Environment(loader=FileSystemLoader(dir_name, encoding="utf8"))
    env.globals["image_file_to_base64"] = _image_file_to_base64
    return env


def _image_file_to_base64(file_path: str):
    with open(file_path, "rb") as image_file:
        data = base64.b64encode(image_file.read())
    return data.decode("utf-8")


class ColorPrint:
    BLACK = "\033[30m"  # (文字)黒
    RED = "\033[31m"  # (文字)赤
    GREEN = "\033[32m"  # (文字)緑
    YELLOW = "\033[33m"  # (文字)黄
    BLUE = "\033[34m"  # (文字)青
    MAGENTA = "\033[35m"  # (文字)マゼンタ
    CYAN = "\033[36m"  # (文字)シアン
    WHITE = "\033[37m"  # (文字)白
    COLOR_DEFAULT = "\033[39m"  # 文字色をデフォルトに戻す
    BOLD = "\033[1m"  # 太字
    UNDERLINE = "\033[4m"  # 下線
    INVISIBLE = "\033[08m"  # 不可視
    REVERCE = "\033[07m"  # 文字色と背景色を反転
    BG_BLACK = "\033[40m"  # (背景)黒
    BG_RED = "\033[41m"  # (背景)赤
    BG_GREEN = "\033[42m"  # (背景)緑
    BG_YELLOW = "\033[43m"  # (背景)黄
    BG_BLUE = "\033[44m"  # (背景)青
    BG_MAGENTA = "\033[45m"  # (背景)マゼンタ
    BG_CYAN = "\033[46m"  # (背景)シアン
    BG_WHITE = "\033[47m"  # (背景)白
    BG_DEFAULT = "\033[49m"  # 背景色をデフォルトに戻す
    RESET = "\033[0m"  # 全てリセット

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


class History:
    start_at: int
    data: dict = {}
    is_function_optimized: bool = False
    success_dict = dict()
    base_dir_name: str

    def __init__(self, columns: list, base_dir_name: str):
        for col in columns:
            self.data[col] = list()
        self.base_dir_name = base_dir_name

    def save(self, col: str, value):
        self.data[col].append(value)

    def start_execution(self):
        self.start_at = time.time()

    def terminated(self, folder_name: str):
        execution_time = time.time() - self.start_at
        self.success_dict[str(len(self.success_dict) + 1)] = bool(
            self.is_function_optimized
        )
        with open(
            os.path.join(self.base_dir_name, folder_name, "setting.txt"), mode="w"
        ) as f:
            f.write(
                f"{str(int(execution_time / 3600))}h-{str(int((execution_time % 3600) / 60))}m-{str(int(execution_time % 60))}s\n"
            )
            f.write(
                f"{'Optimized' if self.is_function_optimized else 'Not Optimized'}\n"
            )
        for col in self.data:
            np.savetxt(
                os.path.join(self.base_dir_name, folder_name, "{}.csv".format(col)),
                np.array(self.data[col]),
                delimiter=",",
                fmt="%.10f",
            )
        self._clear()

    def function_optimized(self):
        self.is_function_optimized = True

    def end(self):
        sr = (
            sum(self.success_dict.values()) / len(self.success_dict)
            if not len(self.success_dict) == 0
            else 0.0
        )
        self.success_dict["SR"] = round(sr, 2) * 100
        with open(os.path.join(self.base_dir_name, "success_list.csv"), mode="w") as f:
            for i, is_suc in self.success_dict.items():
                f.write(f"{i} {is_suc}\n")

    def _clear(self):
        self.start_at = 0
        self.is_function_optimized = False
        for col in self.data:
            self.data[col] = list()
