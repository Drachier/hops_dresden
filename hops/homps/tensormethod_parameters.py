from __future__ import annotations
from enum import Enum
from numbers import Number
from dataclasses import dataclass

import binfootprint as bf
from beartype import beartype

from ..core.hierarchy_parameters import HIParams

class MPSIntegrationMode(Enum):
    TDVP1SITE = "TDVP1"
    TDVP2SITE = "TDVP2"
    TEBD = "TEBD"
    RUNGEKUTTA = "RK"

def generate_parameters(mode: MPSIntegrationMode, **kwargs) -> bf.ABCParameter:
    r"""
    Generates the parameters for the given mode of integration.

    :param mode: The mode of integration.
        Possible modes are:
        - TDVP1SITE: Single site time-dependent variational principle.
        - TDVP2SITE: Two site time-dependent variational principle.
        - TEBD: Time-evolving block decimation.
        - RUNGEKUTTA: Runge-Kutta integration.
    """
    if mode == MPSIntegrationMode.TDVP1SITE:
        return TDVP1SiteParameters(**kwargs)
    elif mode == MPSIntegrationMode.TDVP2SITE:
        return TDVP2SiteParameters(**kwargs)
    elif mode == MPSIntegrationMode.TEBD:
        return TEBDParameters(**kwargs)
    elif mode == MPSIntegrationMode.RUNGEKUTTA:
        raise NotImplementedError("Runge-Kutta integration is not yet implemented.")
    else:
        raise ValueError(f"Invalid mode {mode}.")

def positivity_test(value: Number, custom_name: str = "number"):
    r"""
    Tests if the given value is positive. If it is not, a value error is raised.

    :param value: The value to test.
    :param custom_name: The name of the value to be used in the error message.
    """
    if value <= 0:
        errstr = custom_name + " must be positive!"
        raise ValueError(errstr)

class TDVP1SiteParameters(bf.ABCParameter):
    r"""
    Sets the parameters for the single site time-dependent variational principle.

    :param numiter_lanczos: The number of iterations for the Lanczos algorithm to use.
    :param max_bond_dimension: The maximum bond dimension possible for the MPS.
    """

    __slots__ = ["numiter_lanczos", "max_bond_dimension"]

    @beartype
    def __init__(self,
                 numiter_lanczos: int,
                 max_bond_dimension: int):
        positivity_test(numiter_lanczos, "number of Lanczos iterations")
        positivity_test(max_bond_dimension, "maximum bond dimension")
        self.numiter_lanczos = numiter_lanczos
        self.max_bond_dimension = max_bond_dimension

class TDVP2SiteParameters(bf.ABCParameter):
    r"""
    Sets the parameters for the two site time-dependent variational principle.

    :param numiter_lanczos: The number of iterations for the Lanczos algorithm to use.
    :param max_bond_dimension: The maximum bond dimension possible for the MPS.
    :param relative_svd_tolerance: The relative tolerance for the singular value decomposition. Smaller singular values will be truncated.
    """

    __slots__ = ["numiter_lanczos", "max_bond_dimension", "relative_svd_tolerance"]

    @beartype
    def __init__(self,
                 numiter_lanczos: int,
                 max_bond_dimension: int,
                 relative_svd_tolerance: float):
        positivity_test(numiter_lanczos, "number of Lanczos iterations")
        positivity_test(max_bond_dimension, "maximum bond dimension")
        positivity_test(relative_svd_tolerance, "relative SVD tolerance")
        self.numiter_lanczos = numiter_lanczos
        self.max_bond_dimension = max_bond_dimension
        self.relative_svd_tolerance = relative_svd_tolerance

class TEBDParameters(bf.ABCParameter):
    r"""
    Sets the parameters for the time-evolving block decimation.

    :param max_bond_dimension: The maximum bond dimension possible for the MPS.
    :param svd_relative_tolerance: The relative tolerance for the singular value decomposition. Smaller singular values will be truncated.
    """

    __slots__ = ["max_bond_dimension", "svd_relative_tolerance"]

    @beartype
    def __init__(self,
                 max_bond_dimension: int,
                 svd_relative_tolerance: float):
        positivity_test(max_bond_dimension, "maximum bond dimension")
        positivity_test(svd_relative_tolerance, "SVD relative tolerance")
        self.max_bond_dimension = max_bond_dimension
        self.svd_relative_tolerance = svd_relative_tolerance

@beartype
@dataclass
class HIParamsWTensors(HIParams):
    r"""
    Enhances the container for the parameters of the hierarchy integrator with tensor network parameters.
    """

    __slots__ = ["TensP"]

    TensP: bf.ABCParameter
    """The parameters for the tensor network method."""
