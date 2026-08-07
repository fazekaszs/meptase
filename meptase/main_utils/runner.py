from typing import ClassVar
from dataclasses import dataclass

from ..exceptions import DeserializationException


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
