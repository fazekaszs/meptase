import json
import logging

from pathlib import Path
from argparse import ArgumentParser, Namespace

from .main_utils.io import IOControl, io_main
from .collective_variables import MergeCV
from .additional_potentials import MergedPEF
from .main_utils.deserialize import (
    deserialize_cvs, deserialize_additional_potentials,
    deserialize_kernels, deserialize_calculator,
    deserialize_run_control
)
from .main_utils.diagnostics import (
    cv_mappers_log, additional_potentials_log,
    rdkit_mol_log
)
from .main_utils.runner import runner_main

logger = logging.getLogger(__name__)


def parse_arguments() -> Namespace:

    parser = ArgumentParser()
    parser.add_argument(
        "-in", "--input_file",
        type=Path, required=True, help="The input json file."
    )
    parser.add_argument(
        "-ll", "--log_level",
        type=str, required=False, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Sets the logging level."
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:

    log_formatter = logging.Formatter(
        "[{levelname} | {name} | {asctime}] {message}",
        style="{", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)


def main():

    # Parse the CLI arguments and load the JSON dictionary
    arguments = parse_arguments()
    with open(arguments.input_file, "r") as f:
        input_file_content = json.load(f)

    # Setup logging
    setup_logging(arguments.log_level)

    # Perform input operations: read in IO config, read in the molecule and its attributes
    io_control = IOControl(**input_file_content["io_control"])
    rdkit_mol, ase_mol, map_num_to_idx = io_main(io_control)

    # Log useful information about the molecule
    rdkit_mol_log(rdkit_mol)

    # Deserialize and collect the CVs from the JSON file
    cv_mappers = deserialize_cvs(input_file_content["collective_variables"], map_num_to_idx)

    # Run the CVs as a test on the current coordinates
    cv_mappers_log(ase_mol, cv_mappers)

    # Merge the CV collection to a single CV
    merged_cvs = MergeCV("cv_merger", cv_mappers)

    # Deserialize and collect the PEFs from the JSON file
    all_additional_potentials = deserialize_additional_potentials(
        input_file_content["additional_potentials"],
        merged_cvs
    )

    # Run the PEFs as a test on the current CVs
    additional_potentials_log(ase_mol, all_additional_potentials, merged_cvs)

    # Merge the PEFs to a single PEF
    merged_pef = MergedPEF(all_additional_potentials)

    # Deserialize and collect the kernels from the JSON file
    all_kernels, kernel_target_cv_indices = deserialize_kernels(
        input_file_content["kernels"],
        merged_cvs
    )

    # Deserialize the unbiased calculator from the JSON file
    unbiased_calculator = deserialize_calculator(input_file_content["unbiased_calculator"])

    # Parse the run control parameters from the JSON file
    run_control = deserialize_run_control(input_file_content["run_control"])

    runner_main(
        ase_mol,
        rdkit_mol,
        merged_cvs,
        merged_pef,
        all_kernels,
        unbiased_calculator,
        kernel_target_cv_indices,
        run_control,
        io_control
    )

if __name__ == "__main__":
    main()
