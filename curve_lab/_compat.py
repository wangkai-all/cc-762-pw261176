"""numpy 2.x 兼容：梯形积分在 numpy 2 中由 trapz 更名为 trapezoid。"""

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz
