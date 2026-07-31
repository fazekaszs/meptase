from typing import Callable

from ase.calculators.calculator import Calculator


CALCULATOR_REGISTRY: dict[str, Callable[..., Calculator]] = dict()


def _register_calculator(
    name: str
) -> Callable[[Callable[..., Calculator]], Callable[..., Calculator]]:
    """
    Creates a decorator that registers a calculator factory function in the CALCULATOR_REGISTRY dictionary.
    This will be later used for the deserialization of calculators from the JSON config.

    :param name: The serialized name of the calculator.
    :return: The decorator that registers the calculator factory.
    """

    def decorator(factory: Callable[..., Calculator]) -> Callable[..., Calculator]:
        CALCULATOR_REGISTRY[name] = factory
        return factory

    return decorator


@_register_calculator("tblite")
def create_tblite(**kwargs) -> Calculator:
    from tblite.ase import TBLite
    return TBLite(**kwargs)


@_register_calculator("mace")
def create_mace(**kwargs) -> Calculator:
    from mace.calculators import MACECalculator
    return MACECalculator(**kwargs)
