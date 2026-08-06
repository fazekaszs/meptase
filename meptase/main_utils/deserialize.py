from typing import Any, ClassVar
from dataclasses import dataclass

from ..exceptions import InvalidTypeSelectionException, DeserializationException
from ..collective_variables import CV_REGISTRY, DeserializableCV, MergeCV
from ..additional_potentials import PEF_REGISTRY, DeserializablePEF
from ..kernels import KERNEL_REGISTRY, DeserializableKernel
from ..calculators import CALCULATOR_REGISTRY, Calculator


@dataclass
class RunControl:

    # Fields annotated as ClassVar will be automatically recognized by the dataclasses
    # decorator and will be created as a class-level variable.
    _positive_fields: ClassVar[tuple[str, ...]] = (
        "temperature", "timestep", "friction", "kernel_height",
        "steps_between_hills", "n_hills", "trajectory_write_interval",
    )

    temperature: float
    timestep: float
    friction: float
    kernel_height: float

    steps_between_hills: int
    n_hills: int
    trajectory_write_interval: int

    def __post_init__(self):

        for field_name in self._positive_fields:
            field_value = getattr(self, field_name)
            if field_value <= 0:
                raise DeserializationException(
                    f"The field \"{field_name}\" in a RunControl object "
                    f"must be positive! Instead, it was set to be {field_value}."
                )


def deserialize_cvs(
    serialized_cvs: list[dict[str, Any]],
    index_mapper: dict[int, int]
) -> list[DeserializableCV]:

    cv_mappers = list()
    for cv_dict in serialized_cvs:

        cv_type = cv_dict["type"]
        if cv_type not in CV_REGISTRY:
            raise InvalidTypeSelectionException(
                f"The collective variable type {cv_type} is not registered!"
            )

        cv_mappers.append(CV_REGISTRY[cv_type].from_config(
            index_mapper=index_mapper,
            **cv_dict["parameters"]
        ))

    return cv_mappers


def deserialize_additional_potentials(
    serialized_potentials: list[dict[str, Any]],
    merged_cvs: MergeCV
) -> list[DeserializablePEF]:

    all_additional_potentials = list()
    for pef_dict in serialized_potentials:

        pef_type = pef_dict["type"]
        if pef_type not in PEF_REGISTRY:
            raise InvalidTypeSelectionException(
                f"The potential energy function type {pef_type} is not registered!"
            )

        all_additional_potentials.append(PEF_REGISTRY[pef_type].from_config(
            names_to_idx=merged_cvs.names_to_idx,
            cv_names=pef_dict["target_cvs"],
            **pef_dict["parameters"]
        ))

    return all_additional_potentials


def deserialize_kernels(
    serialized_kernels: list[dict[str, Any]],
    merged_cvs: MergeCV
) -> tuple[list[DeserializableKernel], list[int]]:

    all_kernels = list()
    kernel_target_cv_indices: list[int | None] = [None for _ in merged_cvs.names_to_idx]
    for kernel_idx, kernel_dict in enumerate(serialized_kernels):

        kernel_type = kernel_dict["type"]
        if kernel_type not in KERNEL_REGISTRY:
            raise InvalidTypeSelectionException(
                f"The kernel type {kernel_type} is not registered!"
            )

        current_target_cv_names = kernel_dict["target_cvs"]
        for cv_name in current_target_cv_names:
            cv_idx = merged_cvs.names_to_idx[cv_name]
            kernel_target_cv_indices[cv_idx] = kernel_idx

        current_kernel = KERNEL_REGISTRY[kernel_type].from_config(**kernel_dict["parameters"])
        all_kernels.append(current_kernel)

    # Check against unassigned CVs
    kernelless_cvs = [
        position for position, cv_idx in enumerate(kernel_target_cv_indices)
        if cv_idx is None
    ]
    if len(kernelless_cvs) > 0:
        raise DeserializationException(
            "No kernels were found for some of the collective variables! "
            "The config file should provide a kernel for all CVs! "
        )

    return all_kernels, kernel_target_cv_indices


def deserialize_calculator(
    serialized_calculator: dict[str, Any]
) -> Calculator:

    calc_type = serialized_calculator["type"]
    if calc_type not in CALCULATOR_REGISTRY:
        raise InvalidTypeSelectionException(
            f"The calculator type {calc_type} is not registered!"
        )
    unbiased_calculator = CALCULATOR_REGISTRY[calc_type](**serialized_calculator["parameters"])
    return unbiased_calculator


def deserialize_run_control(
    serialized_run_control: dict[str, Any]
) -> RunControl:
    return RunControl(**serialized_run_control)