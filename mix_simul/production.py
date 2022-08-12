import numpy as np
import cvxpy as cp


class PowerSupply:
    def __init__(self):
        pass

    def power(self):
        pass


class IntermittentArray(PowerSupply):
    def __init__(self, potential: np.ndarray, units_per_region: np.ndarray):

        self.potential = potential
        self.units_per_region = units_per_region

    def power(self):
        return np.einsum("ijk,ik", self.potential, self.units_per_region)


class DispatchableArray(PowerSupply):
    def __init__(self, dispatchable: np.ndarray):
        self.dispatchable = np.array(dispatchable)
        self.n_dispatchable_sources = self.dispatchable.shape[0]

    def power(self, gap: np.ndarray) -> np.ndarray:
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
