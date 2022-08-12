import numpy as np 
import numba
from typing import Tuple

@numba.njit
def storage_iterate(dE, capacity, efficiency, n):
    storage = np.zeros(n)

    for i in np.arange(1, n):
        if dE[i] >= 0:
            dE[i] *= efficiency

        storage[i] = np.maximum(0, np.minimum(capacity, storage[i-1]+dE[i-1]))

    return storage

class StorageModel:
    def __int__(self):
        pass


class MultiStorageModel(StorageModel):
    def __init__(self,
        storage_capacities: np.ndarray,
        storage_max_loads=np.ndarray,
        storage_max_deliveries=np.ndarray,
        storage_efficiencies=np.ndarray
    ):

        self.storage_max_loads = np.array(storage_max_loads)
        self.storage_max_deliveries = np.array(storage_max_deliveries)
        self.storage_capacities = np.array(storage_capacities)*self.storage_max_loads
        self.storage_efficiencies = np.array(storage_efficiencies)

        self.n_storages = len(self.storage_capacities)

        assert len(storage_max_loads) == self.n_storages
        assert len(storage_max_deliveries) == self.n_storages
        assert len(storage_efficiencies) == self.n_storages

    
    def run(self, power_delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(power_delta)

        available_power = power_delta
        excess_power = np.maximum(0, available_power)
        deficit_power = np.maximum(0, -available_power)

        storage_try_load = np.zeros((self.n_storages, T))
        storage_try_delivery = np.zeros((self.n_storages, T))

        storage = np.zeros((self.n_storages, T))
        storage_impact = np.zeros((self.n_storages, T))
        dE_storage = np.zeros((self.n_storages, T))

        for i in range(self.n_storages):
            storage_try_load[i] = np.minimum(excess_power, self.storage_max_loads[i])
            storage_try_delivery[i] = np.minimum(deficit_power, self.storage_max_deliveries[i])

            dE_storage[i] = storage_try_load[i]-storage_try_delivery[i]

            storage[i] = storage_iterate(
                dE_storage[i], self.storage_capacities[i], self.storage_efficiencies[i], T
            )

            # impact of storage on the available power        
            storage_impact[i] = -np.diff(storage[i], append=0)
            storage_impact[i] = np.multiply(
                storage_impact[i],
                np.where(storage_impact[i] < 0, 1/self.storage_efficiencies[i], 1)
            )

            available_power += storage_impact[i]
            excess_power = np.maximum(0, available_power)
            deficit_power = np.maximum(0, -available_power)

        return storage, storage_impact