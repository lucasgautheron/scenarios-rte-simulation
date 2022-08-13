import numpy as np
import cvxpy as cp


class PowerSupply:
    def __init__(self):
        pass

    def power(self):
        pass


class IntermittentArray(PowerSupply):
    def __init__(self, load_factors: np.ndarray, capacity_per_region: np.ndarray):
        """Intermittent source model.

        This model is suitable for sources with a fixed load factor,
        typically intermittent renewables.

        :param load_factors: 3D array of shape :math:`S\times T \times R` containing the input load factors.

        S is the amount of sources (solar, onshore and offshore wind, etc.)
        T is the amount of time steps.
        R is the amount of regions.

        :type load_factors: np.ndarray
        :param capacity_per_region: 2D array of shape :math:`S\times R` containing the installed capacity of each type of source within each region, in GW.
        :type capacity_per_region: np.ndarray
        """

        self.load_factors = load_factors
        self.capacity_per_region = capacity_per_region

        assert (
            (self.load_factors >= 0) & (self.load_factors <= 1)
        ).all(), "load factors must be comprised between 0 and 1"

    def power(self) -> np.ndarray:
        """Power production according to the load factors and installed capacity.

        :return: 1D array of size T, containing the power production in GW at each time step.
        :rtype: np.ndarray
        """
        return np.einsum("ijk,ik", self.load_factors, self.capacity_per_region)


class DispatchableArray(PowerSupply):
    def __init__(self, dispatchable: np.ndarray):
        """Dispatchable power supply.

        The model assumes that these sources can be triggered at any time to deliver up to a certain amount of power (the maximum capacity) and up to a certain amount of energy per year.
        Their total power output never exceeds the amount of power needed to meet the demand (no waste).
        The output is determined by minimizing the deficit of power (demand-supply) over the considered time-range.

        :param dispatchable: 2D array of shape :math:`S\times 2` containing the power capacity (in GW) and energy capacity (in GWh) of each of the S sources.
        :type dispatchable: np.ndarray
        """
        self.dispatchable = np.array(dispatchable)
        self.n_dispatchable_sources = self.dispatchable.shape[0]

    def power(self, gap: np.ndarray) -> np.ndarray:
        """Power production optimized for compensating the input gap.

        :param gap: 1D array containing the power gap to compensate in GW at each timestep.
        :type gap: np.ndarray
        :return: 1D array of size T, containing the power production in GW at each time step.
        :rtype: np.ndarray
        """
        # optimize dispatch
        T = len(gap)
        dispatch_power = cp.Variable((self.n_dispatchable_sources, T))

        constraints = (
            [dispatch_power >= 0, cp.sum(dispatch_power, axis=0) <= np.maximum(gap, 0)]
            + [
                dispatch_power[i, :] <= self.dispatchable[i, 0]
                for i in range(self.n_dispatchable_sources)
            ]
            + [
                cp.sum(dispatch_power[i]) <= self.dispatchable[i, 1] * T / (365.25 * 24)
                for i in range(self.n_dispatchable_sources)
            ]
        )

        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.pos(gap - cp.sum(dispatch_power, axis=0)))),
            constraints,
        )

        prob.solve(solver=cp.ECOS, max_iters=300)
        dp = dispatch_power.value

        return dp
