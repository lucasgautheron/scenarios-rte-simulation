from .consumption import *
from .production import *
from .storage import *

from typing import Tuple


class Scenario:
    def __init__(
        self,
        consumption_model: str = "FittedModel",
        yearly_total: float = None,
        sources: dict = {},
        multistorage: dict = {},
        flexibility_power=0,
        flexibility_time=8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run a mix scenario.

        :param yearly_total: Annual consumption in GWh, for models with fixed annual consumption. defaults to None
        :type yearly_total: _type_, optional
        :param sources: float, defaults to {}
        :type sources: dict, optional
        :param multistorage: _description_, defaults to {}
        :type multistorage: dict, optional
        :param flexibility_power: _description_, defaults to 0
        :type flexibility_power: int, optional
        :param flexibility_time: _description_, defaults to 8
        :type flexibility_time: int, optional
        :return: Performance of the scenario.
        :rtype: Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        self.yearly_total = yearly_total
        self.sources = sources
        self.multistorage = multistorage
        self.flexibility_power = flexibility_power
        self.flexibility_time = flexibility_time
        self.consumption_model_name = consumption_model
        pass

    def build_consumption_model(self):
        if self.consumption_model_name == "ThermoModel":
            mdl = ThermoModel
        elif self.consumption_model_name == "FittedModel":
            mdl = FittedConsumptionModel
        else:
            raise NotImplementedError(
                f"consumption model {self.consumption_model_name} not supported"
            )
        
        self.consumption_model = mdl(yearly_total=self.yearly_total)

    def run(self, times, intermittent_load_factors, consumption_model=None):
        # consumption
        if consumption_model is not None:
            self.consumption_model = consumption_model
        else:
            self.build_consumption_model()

        load = self.consumption_model.get(times)

        # intermittent sources (or sources with fixed load factors)
        intermittent_array = IntermittentArray(
            intermittent_load_factors, np.transpose([self.sources["intermittent"]])
        )
        power = intermittent_array.power()

        # adjust power to load with storage
        storage_model = MultiStorageModel(
            np.array(self.multistorage["capacity"])
            * np.array(self.multistorage["power"]),
            self.multistorage["power"],
            self.multistorage["power"],
            self.multistorage["efficiency"],
        )

        storage, storage_impact = storage_model.run(power - load)
        gap = load - power - storage_impact.sum(axis=0)

        if self.flexibility_power > 0:
            flexibility_model = ConsumptionFlexibilityModel(
                self.flexibility_power, self.flexibility_time
            )
            load = flexibility_model.run(load, power + storage_impact.sum(axis=0))

        gap = load - power - storage_impact.sum(axis=0)

        # further adjust power to load with dispatchable power sources
        dispatchable_model = DispatchableArray(self.sources["dispatchable"])
        dp = dispatchable_model.power(gap)

        gap -= dp.sum(axis=0)
        S = np.maximum(gap, 0).mean()

        return S, load, power, gap, storage, dp
