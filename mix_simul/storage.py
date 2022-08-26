import numpy as np
import numba
from typing import Tuple


@numba.njit
def storage_iterate(dE, capacity, efficiency, n):
    storage = np.zeros(n)
    dE[dE >= 0] *= efficiency

    for i in np.arange(1, n):
        storage[i] = np.maximum(0, np.minimum(capacity, storage[i - 1] + dE[i - 1]))

    return storage


class StorageModel:
    def __int__(self):
        pass


class MultiStorageModel(StorageModel):
    def __init__(
        self,
        storage_capacities: np.ndarray,
        storage_max_loads: np.ndarray,
        storage_max_deliveries: np.ndarray,
        storage_efficiencies: np.ndarray,
    ):
        """Multiple storage model.

        The model fills each storage capacity by order of priority when the power supply exceeds demand.
        It then drains each storage capacity according to the same order of priority when the demand exceeds the supply.

        :param storage_capacities: 1D array containing the storage capacity of each storage in GWh. The order of the values define the priority order (lower indices = higher priority, typically batteries which absorb diurnal fluctuations).
        :type storage_capacities: np.ndarray
        :param storage_max_loads: 1D array containing the storage load capacity of each storage in GW.
        :type storage_max_loads: np.ndarray, optional
        :param storage_max_deliveries: 1D array containing the storage output capacity of each storage in GW.
        :type storage_max_deliveries: np.ndarray, optional
        :param storage_efficiencies: 1D array containing the yield of each storage (between 0 and 1).
        :type storage_efficiencies: np.ndarray, optional
        """

        self.storage_max_loads = np.array(storage_max_loads)
        self.storage_max_deliveries = np.array(storage_max_deliveries)
        self.storage_capacities = np.array(storage_capacities)
        self.storage_efficiencies = np.array(storage_efficiencies)

        self.n_storages = len(self.storage_capacities)

        assert len(storage_max_loads) == self.n_storages
        assert len(storage_max_deliveries) == self.n_storages
        assert len(storage_efficiencies) == self.n_storages

        assert (
            (self.storage_efficiencies >= 0) & (self.storage_efficiencies <= 1)
        ).all(), "The efficiency of each storage type must be comprised between 0 and 1"

    def run(self, power_delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the storage model

        :param power_delta: 1D array containing the supply-demand difference in GW for each timestep. Positive values mean supply is higher than demand.
        :type power_delta: np.ndarray
        :return: Tuple of two 2D arrays (each of shape :math:`S\times T`), where S is the amount of storage types and T the amount of timesteps. The first array contains the amount of energy stored in each storage capacity, in GWh. The second array contains the power impact on the grid (negative when storage is loading, positive when storage delivers power).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        T = len(power_delta)

        available_power = power_delta

        storage_try_load = np.zeros((self.n_storages, T))
        storage_try_delivery = np.zeros((self.n_storages, T))

        storage = np.zeros((self.n_storages, T))
        storage_impact = np.zeros((self.n_storages, T))
        dE_storage = np.zeros((self.n_storages, T))

        for i in range(self.n_storages):
            excess_power = np.maximum(0, available_power)
            deficit_power = np.maximum(0, -available_power)

            storage_try_load[i] = np.minimum(excess_power, self.storage_max_loads[i])
            storage_try_delivery[i] = np.minimum(
                deficit_power, self.storage_max_deliveries[i]
            )

            dE_storage[i] = storage_try_load[i] - storage_try_delivery[i]

            storage[i] = storage_iterate(
                dE_storage[i],
                self.storage_capacities[i],
                self.storage_efficiencies[i],
                T,
            )

            # impact of storage on the available power
            storage_impact[i] = -np.diff(storage[i], append=0)
            storage_impact[i] = np.multiply(
                storage_impact[i],
                np.where(storage_impact[i] < 0, 1 / self.storage_efficiencies[i], 1),
            )

            available_power += storage_impact[i]

        return storage, storage_impact
