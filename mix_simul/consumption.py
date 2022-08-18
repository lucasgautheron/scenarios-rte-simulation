import pandas as pd
import numpy as np
import cvxpy as cp
import numba

from statsmodels.formula.api import ols

from typing import Union


class ConsumptionModel:
    def __init__(self, debug: bool = False):
        pass

    def get(self):
        pass


class ThermoModel(ConsumptionModel):
    def __init__(
        self,
        yearly_total: float,
        heat_threshold: float = 15,
        ac_threshold: float = 20,
        debug: bool = False,
    ):
        self.yearly_total = yearly_total
        self.heat_threshold = heat_threshold
        self.ac_threshold = ac_threshold

        # load temperature observations
        self.temperatures = (
            pd.read_parquet("data/temperatures.parquet")
            .groupby("times")["T2M"]
            .mean()
            .sort_index()
        )
        self.temperatures.index = pd.to_datetime(self.temperatures.index, utc=True)
        self.temperatures -= 273.15

        # fit observed consumption to the model
        self.fit(debug=debug)

    def observed_temperatures(
        self, times: Union[pd.Series, np.ndarray], missing_method: str = None
    ) -> np.ndarray:
        times = pd.to_datetime(times, utc=True)
        return self.temperatures.reindex(times, method=missing_method)

    def get(self, times: Union[pd.Series, np.ndarray], temperatures: np.ndarray = None):
        if temperatures is None:
            temperatures = self.observed_temperatures(times, missing_method="nearest")

        times = pd.DataFrame({"time": times}).set_index("time")
        times.index = pd.to_datetime(times.index, utc=True)
        times["h"] = (
            (times.index - self.time_reference).total_seconds() / 3600
        ).astype(int)
        times["Y"] = 1

        for i, f in enumerate(self.frequencies):
            times[f"c_{i+1}"] = (times["h"] * f * 2 * np.pi).apply(np.cos)
            times[f"s_{i+1}"] = (times["h"] * f * 2 * np.pi).apply(np.sin)

        times["heat_offset"] = np.maximum(0, self.heat_threshold - temperatures)
        times["ac_offset"] = np.maximum(0, temperatures - self.ac_threshold)

        for x in ["heat", "ac"]:
            for T in [12, 24]:
                times[f"{x}_c_{T}"] = times[f"{x}_offset"] * (
                    times["h"] * (1 / T) * 2 * np.pi
                ).apply(np.cos)
                times[f"{x}_s_{T}"] = times[f"{x}_offset"] * (
                    times["h"] * (1 / T) * 2 * np.pi
                ).apply(np.sin)

        curve = self.fit_results.predict(times).values

        return curve

    def fit(self, debug: bool = False):
        consumption = pd.read_csv("data/consommation-quotidienne-brute.csv", sep=";")
        consumption["time"] = pd.to_datetime(
            consumption["Date - Heure"].str.replace(":", ""),
            format="%Y-%m-%dT%H%M%S%z",
            utc=True,
        )
        consumption.set_index("time", inplace=True)
        hourly = consumption.resample("1H").mean()

        fit_temperatures = self.observed_temperatures(hourly.index)
        fit_temperatures.dropna(inplace=True)

        hourly = hourly.reindex(fit_temperatures.index)
        hourly["temperature"] = fit_temperatures.loc[hourly.index]

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

        hourly["heat_offset"] = np.maximum(
            0, self.heat_threshold - hourly["temperature"]
        )
        hourly["ac_offset"] = np.maximum(0, hourly["temperature"] - self.ac_threshold)
        components += ["heat_offset", "ac_offset"]

        for x in ["heat", "ac"]:
            for T in [12, 24]:
                hourly[f"{x}_c_{T}"] = hourly[f"{x}_offset"] * (
                    hourly["h"] * (1 / T) * 2 * np.pi
                ).apply(np.cos)
                hourly[f"{x}_s_{T}"] = hourly[f"{x}_offset"] * (
                    hourly["h"] * (1 / T) * 2 * np.pi
                ).apply(np.sin)

                components += [f"{x}_c_{T}", f"{x}_s_{T}"]

        hourly.rename(
            columns={"Consommation brute électricité (MW) - RTE": "conso"}, inplace=True
        )
        hourly["conso"] /= 1000

        # fit load curve to fourier components
        model = ols("conso ~ " + " + ".join(components) + " + C(Y)", data=hourly)
        self.fit_results = model.fit()

        if debug:
            from matplotlib import pyplot as plt

            print(self.fit_results.summary())
            print(self.fit_results.params[-10], self.fit_results.params[-9])

            prediction = self.fit_results.predict(hourly)

            fig, ax = plt.subplots(1, 1)
            ax.plot(hourly.index, prediction, label="model")
            ax.plot(hourly.index, hourly["conso"], label="observed")
            plt.legend()
            plt.show()

        # normalize according to the desired total yearly consumption
        # this is only approximate.
        intercept = (
            self.fit_results.params[0]
            + self.fit_results.params[1]
            + hourly["heat_offset"].mean() * self.fit_results.params[-10]
            + hourly["ac_offset"].mean() * self.fit_results.params[-9]
        )

        self.fit_results.params *= self.yearly_total / (intercept * 365.25 * 24)


class FittedConsumptionModel(ConsumptionModel):
    def __init__(self, yearly_total: float, debug: bool = False):
        """Consumption Model fitted against observations of French electricity consumption (2012-2021).

        The model captures seasonal, weekly and diurnal variations. It has only 1 parameter, the total annual consumption.

        :param yearly_total: Total annual consumption in GWh.
        :type yearly_total: float
        """
        self.yearly_total = yearly_total
        self.fit()

    def get(self, times: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Retrieve the consumption for each timestamp from the input array.

        :param times: 1D array containing the input timestamps
        :type times: Union[pd.Series, np.ndarray]
        :return: 1D array of floats (consumption in GW) with the same length as the input.
        :rtype: np.ndarray
        """
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
        """Consumption flexibility model.

        Adapts consumption to supply under constraints.

        :param flexibility_power: Maximum power that can be postponed at any time, in GW
        :type flexibility_power: float
        :param flexibility_time: Maximum consumption delay in hours
        :type flexibility_time: int
        """
        self.flexibility_power = flexibility_power
        self.flexibility_time = flexibility_time

    def run(self, load: np.ndarray, supply: np.ndarray) -> np.ndarray:
        """Runs the model.

        Given initial load and supply, the model returns an adjusted load optimized under constraints.

        :param load: 1D array containing the initial load at each timestep, in GW
        :type load: np.ndarray
        :param supply: 1D array containing the power supply at each timestep, in GW
        :type supply: np.ndarray
        :return: 1D array containing the adjusted load at each timestep, in GW
        :rtype: np.ndarray
        """
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
