from .consumption import *
from .production import *
from .storage import *

import yaml


class Scenario:
    def __init__(
        self,
        yearly_total=645 * 1000,
        sources: dict = {},
        multistorage: dict = {},
        flexibility_power=0,
        flexibility_time=8,
    ):
        self.yearly_total = yearly_total
        self.sources = sources
        self.multistorage = multistorage
        self.flexibility_power = flexibility_power
        self.flexibility_time = flexibility_time
        pass

    def run(self, times, intermittent_load_factors):
        # consumption
        consumption_model = FittedConsumptionModel(self.yearly_total)
        load = consumption_model.get(times)

        # intermittent sources (or sources with fixed load factors)
        intermittent_array = IntermittentArray(
            intermittent_load_factors, np.transpose([self.sources["intermittent"]])
        )
        power = intermittent_array.power()

        if self.flexibility_power > 0:
            flexibility_model = ConsumptionFlexibilityModel(
                self.flexibility_power, self.flexibility_time
            )
            load = flexibility_model.run(load, power)

        power_delta = power - load

        # adjust power to load with storage
        storage_model = MultiStorageModel(
            self.multistorage["capacity"],
            self.multistorage["power"],
            self.multistorage["power"],
            self.multistorage["efficiency"],
        )

        storage, storage_impact = storage_model.run(power_delta)
        gap = load - power - storage_impact.sum(axis=0)

        # further adjust power to load with dispatchable power sources
        dispatchable_model = DispatchableArray(self.sources["dispatchable"])
        dp = dispatchable_model.power(gap)

        gap -= dp.sum(axis=0)
        S = np.maximum(gap, 0).mean()

        return S, load, power, gap, storage, dp
