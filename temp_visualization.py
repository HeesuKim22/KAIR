import numpy as np
from imwrite import imwrite
import torch

# arr = np.arange(3 * 3 * 3)
# arr = np.reshape(arr, [3, 3, 3])

# imwrite(
#     arr,
# )

tensor = torch.arange(3 * 3 * 3)
tensor = tensor / tensor.max()
tensor = tensor.reshape([3, 3, 3], tensor)

imwrite(
    tensor,
)
