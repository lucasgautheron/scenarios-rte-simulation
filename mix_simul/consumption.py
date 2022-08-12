import pandas as pd
import numpy as np
import cvxpy as cp

from statsmodels.formula.api import ols

from typing import Union


class ConsumptionModel:
    def __init__(self):
        pass

    def get(self):
        pass


class ThermoModel(ConsumptionModel):
    def __init__(self):
        pass

    def get(temperatures: np.ndarray):
        pass


class FittedConsumptionModel(ConsumptionModel):
    def __init__(self, yearly_total: float):
        self.yearly_total = yearly_total
        self.fit()

    def get(self, times: Union[pd.Series, np.ndarray]):
        # compute the load for the desired timestamps
        times = pd.DataFrame({"time": times}).set_index("time")
        times.index = pd.to_datetime(times.index, utc=True)
        times["h"] = (
            (times.index - self.time_reference).total_seconds() / 3600
        ).astype(int)
        times["Y"] = 1
        for i, f in enumerate(self.frequencies):
            times[f"c_{i+1}"] = (times["h"] * f * 2 * np.pi).apply(np.cos)
            times[f"s_{i+1}"] = (times["h"] * f * 2 * np.pi).apply(np.sin)

        curve = self.fit_results.predict(times).values
        return curve

    def fit(self):
        consumption = pd.read_csv("data/consommation-quotidienne-brute.csv", sep=";")
        consumption["time"] = pd.to_datetime(
            consumption["Date - Heure"].str.replace(":", ""),
            format="%Y-%m-%dT%H%M%S%z",
            utc=True,
        )
        consumption.set_index("time", inplace=True)
        hourly = consumption.resample("1H").mean()

        self.time_reference = hourly.index.min()

        hourly["h"] = (
            (hourly.index - self.time_reference).total_seconds() / 3600
        ).astype(int)

        hourly["Y"] = hourly.index.year - hourly.index.year.min()

        # generate fourier components
        self.frequencies = (
            list((1 / (365.25 * 24)) * np.arange(1, 13))
            + list((1 / (24 * 7)) * np.arange(1, 7))
            + list((1 / 24) * np.arange(1, 12))
        )
        components = []

        for i, f in enumerate(self.frequencies):
            hourly[f"c_{i+1}"] = (hourly["h"] * f * 2 * np.pi).apply(np.cos)
            hourly[f"s_{i+1}"] = (hourly["h"] * f * 2 * np.pi).apply(np.sin)

            components += [f"c_{i+1}", f"s_{i+1}"]

        hourly.rename(
            columns={"Consommation brute électricité (MW) - RTE": "conso"}, inplace=True
        )
        hourly["conso"] /= 1000

        # fit load curve to fourier components
        model = ols("conso ~ " + " + ".join(components) + " + C(Y)", data=hourly)
        self.fit_results = model.fit()

        # normalize according to the desired total yearly consumption
        intercept = self.fit_results.params[0] + self.fit_results.params[1]
        self.fit_results.params *= self.yearly_total / (intercept * 365.25 * 24)


class ConsumptionFlexibilityModel:
    def __init__(self, flexibility_power: float, flexibility_time: int):
        self.flexibility_power = flexibility_power
        self.flexibility_time = flexibility_time

    def run(self, load: np.ndarray, supply: np.ndarray):
        from functools import reduce

        T = len(supply)
        tau = self.flexibility_time

        h = cp.Variable((T, tau + 1))

        constraints = [
            h >= 0,
            h <= 1,
            # bound on the fraction of the demand at any time that can be postponed
            h[:, 0] >= 1 - self.flexibility_power / load,
            cp.multiply(load, cp.sum(h, axis=1)) - load <= self.flexibility_power,
        ] + [
            # total demand conservation
            reduce(lambda x, y: x + y, [h[t - l, l] for l in range(tau)]) == 1
            for t in np.arange(tau, T)
        ]

        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.pos(cp.multiply(load, cp.sum(h, axis=1)) - supply))),
            constraints,
        )

        prob.solve(verbose=True, solver=cp.ECOS, max_iters=300)

        hb = np.array(h.value)
        return load * np.sum(hb, axis=1)
