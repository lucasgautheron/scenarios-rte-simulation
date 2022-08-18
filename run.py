import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import argparse
import yaml

from mix_simul.scenarios import Scenario

parser = argparse.ArgumentParser()
parser.add_argument(
    "--begin",
    help="begin date (YYYY-MM-DD), between 1985-01-01 and 2015-01-01",
    required=True,
)
parser.add_argument(
    "--end",
    help="end date (YYYY-MM-DD), between 1985-01-01 and 2015-01-01",
    required=True,
)
parser.add_argument(
    "--flexibility", help="enable load flexibility modeling", action="store_true"
)
parser.add_argument(
    "--scenarios",
    help="path to scenarios parameters yml file",
    default="scenarios/rte_2050.yml",
)
args = parser.parse_args()

with open(args.scenarios, "r") as f:
    scenarios = yaml.load(f, Loader=yaml.FullLoader)

potential = pd.read_parquet("data/potential.parquet")
potential = potential.loc["1985-01-01 00:00:00":"2015-01-01 00:00:00", :]
potential.fillna(0, inplace=True)

begin = args.begin
end = args.end
flexibility = args.flexibility

potential = potential.loc[(slice(f"{begin} 00:00:00", f"{end} 00:00:00"), "FR"), :]

# intermittent sources potential
p = potential[["onshore", "offshore", "solar"]].to_xarray().to_array()
p = np.insert(p, 3, 0.7, axis=0)  # nuclear power-like

times = potential.index.get_level_values(0)

n_scenarios = len(scenarios)

fig, axes = plt.subplots(nrows=n_scenarios, ncols=2, sharex="col", sharey=True)
w, h = fig.get_size_inches()
fig.set_figwidth(w * 1.5)
fig.set_figheight(h * 1.5)

fig_storage, axes_storage = plt.subplots(
    nrows=n_scenarios, ncols=2, sharex="col", sharey=True
)
fig_storage.set_figwidth(w * 1.5)
fig_storage.set_figheight(h * 1.5)

fig_dispatch, axes_dispatch = plt.subplots(
    nrows=n_scenarios, ncols=2, sharex="col", sharey=True
)
fig_dispatch.set_figwidth(w * 1.5)
fig_dispatch.set_figheight(h * 1.5)

fig_gap_distribution, axes_gap_distribution = plt.subplots(
    nrows=int(np.ceil(n_scenarios / 2)), ncols=2, sharex="col", sharey=True
)
fig_gap_distribution.set_figwidth(w * 1.5)
fig_gap_distribution.set_figheight(h * 1.5)

months = ["Février", "Juin"]

labels = [
    "load (GW)",
    "production (GW)",
    "available power (production-storage) (GW)",
    "power deficit",
]

labels_storage = ["Batteries (TWh)", "STEP (TWh)", "P2G (TWh)"]

labels_dispatch = ["Hydro (GW)", "Biomass (GW)", "Thermal (GW)"]

date_fmt = mdates.DateFormatter("%d/%m")

row = 0
previous_consumption_model = None

for scenario in scenarios:
    if not flexibility:
        scenarios[scenario]["flexibility_power"] = 0

    scenario_model = Scenario(**scenarios[scenario])
    S, load, production, gap, storage, dp = scenario_model.run(
        times, p, consumption_model=previous_consumption_model
    )
    previous_consumption_model = scenario_model.consumption_model

    print(f"{scenario}:", S, gap.max(), np.quantile(gap, 0.95))
    print(
        f"exports: {np.minimum(np.maximum(-gap, 0), 39).sum()/1000} TWh; imports: {np.minimum(np.maximum(gap, 0), 39).sum()/1000} TWh"
    )
    print(
        f"dispatchable: "
        + ", ".join([f"{dp[i].sum()/1000:.2f} TWh" for i in range(dp.shape[0])])
    )

    potential["load"] = load
    potential["production"] = production
    potential["available"] = production - np.diff(storage.sum(axis=0), append=0)
    potential["gap"] = gap

    for i in range(3):
        potential[f"storage_{i}"] = np.diff(
            storage[i, :], append=0
        )  # storage[i,:]/1000
        potential[f"storage_{i}"] = storage[i, :] / 1000

    for i in range(dp.shape[0]):
        potential[f"dispatch_{i}"] = dp[i, :]

    potential["dispatch"] = dp.sum(axis=0)

    data = [
        potential.loc[(slice("2013-02-01 00:00:00", "2013-03-01 00:00:00"), "FR"), :],
        potential.loc[(slice("2013-06-01 00:00:00", "2013-07-01 00:00:00"), "FR"), :],
    ]

    for col in range(2):
        ax = axes[row, col] if axes.ndim > 1 else axes[col]
        ax.plot(
            data[col].index.get_level_values(0),
            data[col]["load"],
            label="adjusted load (GW)",
            lw=1,
        )
        ax.plot(
            data[col].index.get_level_values(0),
            data[col]["production"],
            label="production (GW)",
            ls="dotted",
            lw=1,
        )
        ax.plot(
            data[col].index.get_level_values(0),
            data[col]["available"],
            label="available power (production-d(storage)/dt) (GW)",
            lw=1,
        )

        ax.fill_between(
            data[col].index.get_level_values(0),
            data[col]["available"],
            data[col]["load"],
            where=data[col]["load"] > data[col]["available"],
            color="red",
            alpha=0.15,
        )

        ax.xaxis.set_major_formatter(date_fmt)
        ax.text(
            0.5,
            0.87,
            f"Scénario {scenario} ({months[col]})",
            ha="center",
            transform=ax.transAxes,
        )

        ax.set_ylim(10, 210)

        ax = axes_storage[row, col] if axes.ndim > 1 else axes_storage[col]
        for i in np.arange(3):
            if i == 2:
                base = 0
            else:
                base = np.sum(
                    [data[col][f"storage_{j}"] for j in np.arange(i + 1, 3)], axis=0
                )

            ax.fill_between(
                data[col].index.get_level_values(0),
                base,
                base + data[col][f"storage_{i}"],
                label=f"storage {i}",
                alpha=0.5,
            )

            ax.plot(
                data[col].index.get_level_values(0),
                base + data[col][f"storage_{i}"],
                label=f"storage {i}",
                lw=0.25,
            )

        ax.xaxis.set_major_formatter(date_fmt)
        ax.text(
            0.5,
            0.87,
            f"Scénario {scenario} ({months[col]})",
            ha="center",
            transform=ax.transAxes,
        )

        ax = axes_dispatch[row, col] if axes.ndim > 1 else axes_dispatch[col]
        for i in range(dp.shape[0]):
            if i == 0:
                base = 0
            else:
                base = np.sum(
                    [data[col][f"dispatch_{j}"] for j in np.arange(i)], axis=0
                )

            ax.fill_between(
                data[col].index.get_level_values(0),
                base,
                base + data[col][f"dispatch_{i}"],
                label=f"dispatch {i}",
                alpha=0.5,
            )

            ax.plot(
                data[col].index.get_level_values(0),
                base + data[col][f"dispatch_{i}"],
                label=f"dispatch {i}",
                lw=0.25,
            )

        ax.xaxis.set_major_formatter(date_fmt)
        ax.text(
            0.5,
            0.87,
            f"Scénario {scenario} ({months[col]})",
            ha="center",
            transform=ax.transAxes,
        )

        div = int(np.ceil(n_scenarios / 2))
        ax = (
            axes_gap_distribution[row % div, row // div]
            if axes_gap_distribution.ndim > 1
            else axes_gap_distribution[row % div]
        )

        hist, bin_edges = np.histogram(-gap, bins=1000)
        hist = np.cumsum(hist)
        hist = 100 * (hist - hist.min()) / hist.ptp()
        keep = np.abs(bin_edges[:-1]) < 50
        ax.plot(
            bin_edges[:-1][keep], hist[keep], lw=1, label="power gap", color="#ff7f00"
        )

        years = pd.date_range(start=begin, end=end, freq="Y")
        for i in range(len(years) - 1):
            year_data = potential.loc[
                (slice(f"{years[i]} 00:00:00", f"{years[i+1]} 00:00:00"), "FR"), "gap"
            ]

            hist, bin_edges = np.histogram(-year_data, bins=1000)
            hist = np.cumsum(hist)
            hist = 100 * (hist - hist.min()) / hist.ptp()
            keep = np.abs(bin_edges[:-1]) < 50
            ax.plot(
                bin_edges[:-1][keep], hist[keep], lw=0.5, alpha=0.2, color="#ff7f00"
            )

        ax.text(0.5, 0.87, f"Scénario {scenario}", ha="center", transform=ax.transAxes)

    row += 1

for axs in [axes, axes_dispatch, axes_storage]:
    if axes.ndim > 1:
        for label in (
            axs[-1, 0].get_xmajorticklabels() + axs[-1, 1].get_xmajorticklabels()
        ):
            label.set_rotation(30)
            label.set_horizontalalignment("right")
    else:
        for label in axs[0].get_xmajorticklabels() + axs[-1].get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

flex = "With" if flexibility else "Without"


def plot_path(name):
    return "output/{}{}.png".format(name, "_flexibility" if flexibility else "")


plt.subplots_adjust(wspace=0, hspace=0)
fig.suptitle(
    f"Simulations based on {begin}--{end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)"
)
fig.text(1, 0, "Lucas Gautheron", ha="right")
fig.legend(
    labels,
    loc="lower right",
    bbox_to_anchor=(1, -0.1),
    ncol=len(labels),
    bbox_transform=fig.transFigure,
)
fig.savefig(plot_path("load_supply"), bbox_inches="tight", dpi=200)

fig_storage.suptitle(
    f"Simulations based on {begin}--{end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)"
)
fig_storage.text(1, 0, "Lucas Gautheron", ha="right")
fig_storage.legend(
    labels_storage,
    loc="lower right",
    bbox_to_anchor=(1, -0.1),
    ncol=len(labels_storage),
    bbox_transform=fig_storage.transFigure,
)
fig_storage.savefig(plot_path("storage"), bbox_inches="tight", dpi=200)

fig_dispatch.suptitle(
    f"Simulations based on {begin}--{end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)"
)
fig_dispatch.text(1, 0, "Lucas Gautheron", ha="right")
fig_dispatch.legend(
    labels_dispatch,
    loc="lower right",
    bbox_to_anchor=(1, -0.1),
    ncol=len(labels_dispatch),
    bbox_transform=fig_dispatch.transFigure,
)
fig_dispatch.savefig(plot_path("dispatch"), bbox_inches="tight", dpi=200)

fig_gap_distribution.suptitle(
    f"Power gap cumulative distribution (%)\nSimulations based on {begin}--{end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)"
)
fig_gap_distribution.legend(
    ["Power gap (available-load) (GW)"],
    loc="lower right",
    bbox_to_anchor=(1, -0.1),
    ncol=1,
    bbox_transform=fig_dispatch.transFigure,
)
fig_gap_distribution.text(1, 0, "Lucas Gautheron", ha="right")
fig_gap_distribution.savefig(
    plot_path("gap_distribution"), bbox_inches="tight", dpi=200
)

plt.show()
