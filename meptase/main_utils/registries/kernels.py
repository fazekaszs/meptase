from typing import Callable
from abc import ABC

from ...kernels import (
    KernelBase,
    GaussianKernel, VonMisesKernel, BetaKernel
)

class DeserializableKernel(KernelBase, ABC):

    @classmethod
    def from_config[T: KernelBase](cls: type[T], **kwargs) -> T:
        return cls(**kwargs)


KERNEL_REGISTRY: dict[str, type[DeserializableKernel]] = dict()


def _register_kernel[T: DeserializableKernel](name: str) -> Callable[[type[T], ], type[T]]:

    def decorator(cls: type[T]) -> type[T]:
        KERNEL_REGISTRY[name] = cls
        return cls

    return decorator


@_register_kernel("gaussian")
class GaussianKernelS(GaussianKernel, DeserializableKernel):
    pass


@_register_kernel("von_mises")
class VonMisesKernelS(VonMisesKernel, DeserializableKernel):
    pass


@_register_kernel("beta")
class BetaKernelS(BetaKernel, DeserializableKernel):
    pass

