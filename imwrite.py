from datetime import datetime

import cv2
import numpy as np
import torch


def imwrite(img):
    if type(img) == np.ndarray:
        arr2img(img)
    elif type(img) == torch.Tensor:
        tensor2img(img)


def arr2img(arr: np.ndarray, filename: str = "") -> None:
    arr_int = np.uint8(arr)

    filename = gen_filename(*arr_int.shape, "img_arr")

    assert arr_int.dtype == "uint8", "img not `uint8`"
    cv2.imwrite(filename, arr_int)


# def tensor2img(tensor: torch.Tensor, filename: str = "") -> None:
#     assert 0 <= tensor.max() <= 1, "tensor greater than 1 or less than 0"
#     tensor = tensor * 255.0
#     arr = tensor.numpy()
#     arr_int = np.uint8(arr)

#     filename = gen_filename(*arr_int.shape, "img_tensor")

#     assert arr_int.dtype == "uint8", "img not `uint8`"
#     cv2.imwrite(filename, arr_int)


def gen_filename(
    W: int,
    H: int,
    C: int,
    prefix: str,
) -> str:
    cur_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    shape_str = [str(W), str(H), str(C)]
    shape_str = "x".join(shape_str)
    filename_str = [prefix, shape_str, cur_time_str]
    filename_str = "_".join(filename_str) + ".jpg"

    return filename_str
