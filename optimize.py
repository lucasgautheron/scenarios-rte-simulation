import pandas as pd
import numpy as np
import numba

from numpy.random import dirichlet
from scipy.optimize import linprog
from scipy.signal import square
from statsmodels.formula.api import ols

import cvxpy as cp

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--begin")
parser.add_argument("--end")
parser.add_argument("--flexibility", action="store_true")
args = parser.parse_args()


def achieved_load_factors(sources=[]):
    # installed capacity
    infra = pd.read_csv(
        "registre-national-installation-production-stockage-electricite-agrege.csv", sep=";")
    infra["Year"] = infra["dateMiseEnService"].str[-4:].astype(str)
    infra.dropna(subset=["Year", "puisMaxInstallee"], inplace=True)
    infra = infra[infra["Year"] != 'nan']
    infra = infra.groupby(["filiere", "Year"]).agg(
        pi=('puisMaxInstallee', 'sum')
    ).reset_index().sort_values(["filiere", "Year"])

    infra['pi'] = infra.groupby(
        'filiere')['pi'].transform(pd.Series.cumsum)/1000
    infra["Year"] = infra["Year"].astype(int)
    infra.set_index(["filiere", "Year"], inplace=True)
    infra = infra.reindex(
        index=[
            (filiere, year) for filiere in set(infra.index.get_level_values(0)) for year in np.arange(2000, 2022)
        ]
    )
    infra = infra.sort_index(level=0).ffill().reindex(infra.index)
    infra.reset_index(inplace=True)
    infra = infra.pivot(index="Year", columns="filiere", values="pi")

    # production
    prod = pd.read_csv("eco2mix-national-cons-def.csv", sep=";")
    prod["time"] = pd.to_datetime(
        prod["Date et Heure"].str.replace(":", ""), format="%Y-%m-%dT%H%M%S%z", utc=True
    )
    prod["Year"] = prod["time"].dt.year
    prod = prod.merge(infra, left_on="Year", right_on="Year", how="left").sort_values("time").set_index("time")

    # consumption
    consumption = pd.read_csv("consommation-quotidienne-brute.csv", sep=";")
    consumption['time'] = pd.to_datetime(
        consumption["Date - Heure"].str.replace(":", ""), format="%Y-%m-%dT%H%M%S%z", utc=True)
    consumption.set_index("time", inplace=True)

    print(consumption["Consommation brute électricité (MW) - RTE"])

    for source in sources:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        prod[f"{source} (load factor)"] = prod[f"{source} (MW)"]/prod[source]
        print(prod[f"{source} (load factor)"])
        ax1.scatter(prod.index, prod[f"{source} (load factor)"], s=0.5, color="red")
        ax2.plot(consumption.index, consumption["Consommation brute électricité (MW) - RTE"], lw=0.5, color="blue")
        plt.show()
        fig.savefig(f"test_{source}.png", dpi=200)
        plt.clf()
        plt.cla()

    # plt.show()


# achieved_load_factors(["Nucléaire", "Hydraulique"])


def generate_load_curve(times, yearly_total=645*1000):
    # load observed load curve
    consumption = pd.read_csv("consommation-quotidienne-brute.csv", sep=";")
    consumption['time'] = pd.to_datetime(
        consumption["Date - Heure"].str.replace(":", ""), format="%Y-%m-%dT%H%M%S%z", utc=True)
    consumption.set_index("time", inplace=True)
    hourly = consumption.resample('1H').mean()
    hourly['h'] = ((hourly.index - hourly.index.min()
                    ).total_seconds()/3600).astype(int)

    hourly['Y'] = hourly.index.year-hourly.index.year.min()

    # generate fourier components
    frequencies = list((1/(365.25*24))*np.arange(1, 13)) + \
        list((1/(24*7)) * np.arange(1, 7)) + \
        list((1/24) * np.arange(1, 12))
    components = []

    for i, f in enumerate(frequencies):
        hourly[f'c_{i+1}'] = (hourly['h']*f*2*np.pi).apply(np.cos)
        hourly[f's_{i+1}'] = (hourly['h']*f*2*np.pi).apply(np.sin)

        components += [f'c_{i+1}', f's_{i+1}']

    hourly.rename(
        columns={'Consommation brute électricité (MW) - RTE': 'conso'}, inplace=True)
    hourly["conso"] /= 1000

    # fit load curve to fourier components
    model = ols("conso ~ " + " + ".join(components)+" + C(Y)", data=hourly)
    results = model.fit()

    # normalize according to the desired total yearly consumption
    intercept = results.params[0]+results.params[1]
    results.params *= yearly_total/(intercept*365.25*24)

    # compute the load for the desired timestamps
    times = pd.DataFrame({'time': times}).set_index("time")
    times.index = pd.to_datetime(times.index, utc=True)
    times['h'] = ((times.index - hourly.index.min()
                   ).total_seconds()/3600).astype(int)
    times['Y'] = 1
    for i, f in enumerate(frequencies):
        times[f'c_{i+1}'] = (times['h']*f*2*np.pi).apply(np.cos)
        times[f's_{i+1}'] = (times['h']*f*2*np.pi).apply(np.sin)

    curve = results.predict(times)
    return np.array(curve)


@numba.njit
def storage_iterate(dE, capacity, efficiency, n):
    storage = np.zeros(n)

    for i in np.arange(1, n):
        if dE[i] >= 0:
            dE[i] *= efficiency

        storage[i] = np.maximum(0, np.minimum(capacity, storage[i-1]+dE[i-1]))

    return storage


def objective_with_storages(
    potential,
    units_per_region,
    dispatchable,
    p_load=2664*1000/(365*24),
    storage_capacities=5*250,
    storage_max_loads=5*30,
    storage_max_deliveries=5*30,
    storage_efficiencies=0.67
):

    power = np.einsum('ijk,ik', potential, units_per_region)
    power_delta = power-p_load

    T = len(power)

    available_power = np.array(power_delta)
    excess_power = np.maximum(0, available_power)
    deficit_power = np.maximum(0, -available_power)

    n_storages = len(storage_capacities)
    storage_try_load = np.zeros((n_storages, T))
    storage_try_delivery = np.zeros((n_storages, T))

    storage = np.zeros((n_storages, T))
    storage_impact = np.zeros((n_storages, T))
    dE_storage = np.zeros((n_storages, T))

    for i in range(n_storages):
        storage_try_load[i] = np.minimum(excess_power, storage_max_loads[i])
        storage_try_delivery[i] = np.minimum(deficit_power, storage_max_deliveries[i])

        dE_storage[i] = storage_try_load[i]-storage_try_delivery[i]

        storage[i] = storage_iterate(
            dE_storage[i], storage_capacities[i], storage_efficiencies[i], T
        )

        # impact of storage on the available power        
        storage_impact[i] = -np.diff(storage[i], append=0)
        storage_impact[i] = np.multiply(
            storage_impact[i],
            np.where(storage_impact[i] < 0, 1/storage_efficiencies[i], 1)
        )

        available_power += storage_impact[i]
        excess_power = np.maximum(0, available_power)
        deficit_power = np.maximum(0, -available_power)

    gap = p_load-power-storage_impact.sum(axis=0)

    # optimize dispatch
    n_dispatchable_sources = dispatchable.shape[0]
    dispatch_power = cp.Variable((n_dispatchable_sources, T))

    constraints = [
        dispatch_power >= 0,
        cp.sum(dispatch_power, axis=0) <= np.maximum(gap, 0)
    ] + [
        dispatch_power[i,:] <= dispatchable[i,:,0].sum()
        for i in range(n_dispatchable_sources)
    ] + [
        cp.sum(dispatch_power[i]) <= dispatchable[i,:,1].sum()
        for i in range(n_dispatchable_sources)
    ]

    prob = cp.Problem(
        cp.Minimize(
            cp.sum(cp.pos(gap-cp.sum(dispatch_power, axis=0))) + cp.max(gap-cp.sum(dispatch_power, axis=0))
        ),
        constraints
    )

    prob.solve(solver=cp.ECOS, max_iters=300)
    dp = dispatch_power.value

    gap -= dp.sum(axis=0)
    S = np.maximum(gap, 0).mean()

    return S, power, gap, storage, dp

def objective_with_storage_and_flexibility(
    potential,
    units_per_region,
    dispatchable,
    p_load=2664*1000/(365*24),
    storage_capacities=5*250,
    storage_max_loads=5*30,
    storage_max_deliveries=5*30,
    storage_efficiencies=0.67,
    flexibility_power=17,
    flexibility_time=8
):

    power = np.einsum('ijk,ik', potential, units_per_region)    

    T = len(power)
    tau = flexibility_time    

    h = cp.Variable((T, tau+1))

    constraints = [
        h >= 0,
        h <= 1,
        h[:, 0] >= 1-flexibility_power/p_load,
        cp.multiply(p_load, cp.sum(h, axis=1))-p_load <= flexibility_power,
    ] + [
        h[t, 0]+h[t-1, 1]+h[t-2, 2]+h[t-3, 3]+h[t-4, 4] +
        h[t-5, 5]+h[t-6, 6]+h[t-7, 7]+h[t-8, 8] == 1
        for t in np.arange(flexibility_time, T)
    ]

    prob = cp.Problem(
        cp.Minimize(
            cp.sum(cp.pos(cp.multiply(p_load, cp.sum(h, axis=1))-power))
        ),
        constraints
    )

    prob.solve(verbose=True, solver=cp.ECOS, max_iters=300)

    hb = np.array(h.value)
    p_load = p_load*np.sum(hb, axis=1)

    power_delta = power-p_load
    available_power = np.array(power_delta)
    excess_power = np.maximum(0, available_power)
    deficit_power = np.maximum(0, -available_power)

    n_storages = len(storage_capacities)
    storage_try_load = np.zeros((n_storages, T))
    storage_try_delivery = np.zeros((n_storages, T))

    storage = np.zeros((n_storages, T))
    storage_impact = np.zeros((n_storages, T))
    dE_storage = np.zeros((n_storages, T))

    for i in range(n_storages):
        storage_try_load[i] = np.minimum(excess_power, storage_max_loads[i])
        storage_try_delivery[i] = np.minimum(deficit_power, storage_max_deliveries[i])

        dE_storage[i] = storage_try_load[i]-storage_try_delivery[i]

        storage[i] = storage_iterate(
            dE_storage[i], storage_capacities[i], storage_efficiencies[i], T
        )

        # impact of storage on the available power        
        storage_impact[i] = -np.diff(storage[i], append=0)
        storage_impact[i] = np.multiply(
            storage_impact[i],
            np.where(storage_impact[i] < 0, 1/storage_efficiencies[i], 1)
        )

        available_power += storage_impact[i]
        excess_power = np.maximum(0, available_power)
        deficit_power = np.maximum(0, -available_power)

    # power missing to meet demand
    gap = p_load-power-storage_impact.sum(axis=0)

    # optimize dispatch
    n_dispatchable_sources = dispatchable.shape[0]
    dispatch_power = cp.Variable((n_dispatchable_sources, T))

    constraints = [
        dispatch_power >= 0,
        cp.sum(dispatch_power, axis=0) <= np.maximum(gap, 0)
    ] + [
        dispatch_power[i,:] <= dispatchable[i,:,0].sum()
        for i in range(n_dispatchable_sources)
    ] + [
        cp.sum(dispatch_power[i]) <= dispatchable[i,:,1].sum()
        for i in range(n_dispatchable_sources)
    ]

    prob = cp.Problem(
        cp.Minimize(
            cp.sum(cp.pos(gap-cp.sum(dispatch_power, axis=0)))
        ),
        constraints
    )

    prob.solve(solver=cp.ECOS, max_iters=300)
    dp = dispatch_power.value

    gap -= dp.sum(axis=0)
    S = np.maximum(gap, 0).mean()

    return S, power, gap, storage, p_load, dp


potential = pd.read_parquet("potential.parquet")
potential = potential.loc['1985-01-01 00:00:00':'2015-01-01 00:00:00', :]
potential.fillna(0, inplace=True)

potential = potential.loc[(
    slice(f'{args.begin} 00:00:00', f'{args.end} 00:00:00'), 'FR'), :]

load = generate_load_curve(potential.index.get_level_values(0))

p = potential[["onshore", "offshore", "solar"]].to_xarray().to_array()
p = np.insert(p, 3, 0.68, axis=0)  # nuclear power-like
# p[-1,:,0] = 0.68+0.25*np.cos(2*np.pi*np.arange(len(potential))/(365.25*24)) # nuclear seasonality ersatz

dispatchable = np.zeros((3, 1, 2))

# hydro power
dispatchable[0][0][0] = 22
dispatchable[0][0][1] = 63*1000*len(potential)/(365.25*24)

# biomass
dispatchable[1][0][0] = 2
dispatchable[1][0][1] = 12*1000*len(potential)/(365.25*24)

# thermique tradi
dispatchable[2][0][0] = 0.5
dispatchable[2][0][1] = 1e8

n_sources = p.shape[0]
n_dt = p.shape[1]
n_regions = p.shape[2]

# start = np.array([18, 0, 10, 61.4, 3])  # current mix
# # stop = np.array([65, 40, 143, 0, 3]) # negawatt like mix
# stop = np.array([74, 62, 208, 0, 3])  # RTE 100% EnR

scenarios = {
    'M0': np.array([74, 62, 208, 0]),
    'M1': np.array([59, 45, 214, 16]),
    'M23': np.array([72, 60, 125, 16]),
    'N1': np.array([58, 45, 118, 16+13]),
    'N2': np.array([52, 36, 90, 16+23]),
    'N03': np.array([43, 22, 70, 24+27]),
}

BATTERY_CAPACITY=4 # 4 hour capacity
STEP_CAPACITY=24
P2G_CAPACITY=24*7*2

battery_power = {
    'M0': 26,
    'M1': 21,
    'M23': 13,
    'N1': 9,
    'N2': 2,
    'N03': 1
}

p2g_power = {
    'M0': 29,
    'M1': 20,
    'M23': 20,
    'N1': 11,
    'N2': 5,
    'N03': 0
}

step_power = 8

fig, axes = plt.subplots(nrows=6, ncols=2, sharex="col", sharey=True)
w, h = fig.get_size_inches()
fig.set_figwidth(w*1.5)
fig.set_figheight(h*1.5)

fig_storage, axes_storage = plt.subplots(nrows=6, ncols=2, sharex="col", sharey=True)
fig_storage.set_figwidth(w*1.5)
fig_storage.set_figheight(h*1.5)

fig_dispatch, axes_dispatch = plt.subplots(nrows=6, ncols=2, sharex="col", sharey=True)
fig_dispatch.set_figwidth(w*1.5)
fig_dispatch.set_figheight(h*1.5)


# for step in np.linspace(start, stop, 2050-2022, True)[::-1]:
row = 0
for scenario in scenarios:
    units = np.transpose([scenarios[scenario]])

    # 8 GW STEP capacity common to all scenarios
    storage_power = battery_power[scenario] + p2g_power[scenario] + step_power
    storage_max_load = battery_power[scenario]*BATTERY_CAPACITY + p2g_power[scenario]*24*7 + step_power*24

    # nuclear_load_model(p[:,:3], units, load, 20)

    if args.flexibility:
        S, production, gap, storage, adjusted_load, dp = objective_with_storage_and_flexibility(
            p, units,
            dispatchable,
            p_load=load,
            storage_capacities=[battery_power[scenario]*BATTERY_CAPACITY, step_power*STEP_CAPACITY, p2g_power[scenario]*P2G_CAPACITY],
            storage_max_deliveries=[battery_power[scenario], step_power, p2g_power[scenario]],
            storage_max_loads=[battery_power[scenario], step_power, p2g_power[scenario]],
            storage_efficiencies=[0.8, 0.8, 0.3]
        )

        print(f"{scenario}:", S, gap.max(), np.quantile(gap, 0.95))
    else:
        S, production, gap, storage, dp = objective_with_storages(
            p,
            units,
            dispatchable,
            p_load = load,
            storage_capacities=[battery_power[scenario]*BATTERY_CAPACITY, step_power*STEP_CAPACITY, p2g_power[scenario]*P2G_CAPACITY],
            storage_max_deliveries=[battery_power[scenario], step_power, p2g_power[scenario]],
            storage_max_loads=[battery_power[scenario], step_power, p2g_power[scenario]],
            storage_efficiencies=[0.8, 0.8, 0.3]
        )
        adjusted_load = load
        print(f"{scenario} w/o flexibility:", S, gap.max(), np.quantile(gap, 0.95))
    
    print(f"exports: {np.minimum(np.maximum(-gap, 0), 39).sum()/1000} TWh; imports: {np.minimum(np.maximum(gap, 0), 39).sum()/1000} TWh")
    print(f"dispatchable: " + ", ".join([f"{dp[i].sum()/1000:.2f} TWh" for i in range(dp.shape[0])]))

    potential['adjusted_load'] = adjusted_load
    potential['production'] = production
    potential['available'] = production-np.diff(storage.sum(axis=0), append=0)

    for i in range(3):
        potential[f"storage_{i}"] = np.diff(storage[i,:], append=0)#storage[i,:]/1000
        potential[f"storage_{i}"] = storage[i,:]/1000

    for i in range(dp.shape[0]):
        potential[f"dispatch_{i}"] = dp[i,:]

    potential["dispatch"] = dp.sum(axis=0)

    data = [
        potential.loc[(slice('2013-02-01 00:00:00',
                       '2013-03-01 00:00:00'), 'FR'), :],
        potential.loc[(slice('2013-06-01 00:00:00',
                       '2013-07-01 00:00:00'), 'FR'), :]
    ]

    months = [
        "Février",
        "Juin"
    ]

    labels = [
        "adjusted load (GW)",
        "production (GW)",
        "available power (production-storage) (GW)",
        "power deficit"
    ]

    labels_storage = [
        "Batterie (TWh)",
        "STEP (TWh)",
        "P2G (TWh)"
    ]

    labels_dispatch = [
        "Hydraulique (GW)",
        "Biomasse (GW)",
        "Thermique gaz (GW)"
    ]

    for col in range(2):
        ax = axes[row, col]
        ax.plot(data[col].index.get_level_values(0), data[col]
                ["adjusted_load"], label="adjusted load (GW)", lw=1)
        ax.plot(data[col].index.get_level_values(0), data[col]
                ["production"], label="production (GW)", ls="dotted", lw=1)
        ax.plot(data[col].index.get_level_values(0), data[col]["available"],
                label="available power (production-d(storage)/dt) (GW)", lw=1)

        ax.fill_between(
            data[col].index.get_level_values(0),
            data[col]["available"],
            data[col]["adjusted_load"],
            where=data[col]["adjusted_load"] > data[col]["available"],
            color='red',
            alpha=0.15
        )

        fmt = mdates.DateFormatter('%d/%m')
        ax.xaxis.set_major_formatter(fmt)
        ax.text(
            0.5, 0.87, f"Scénario {scenario} ({months[col]})", ha='center', transform=ax.transAxes)

        ax.set_ylim(25, 225)

        ax = axes_storage[row, col]
        for i in np.arange(3):
            if i == 2:
                base = 0
            else:
                base = np.sum([data[col][f"storage_{j}"] for j in np.arange(i+1,3)], axis=0)
            
            ax.fill_between(data[col].index.get_level_values(0), base, base+data[col]
                [f"storage_{i}"], label=f"storage {i}", alpha=0.5)

            ax.plot(data[col].index.get_level_values(0), base+data[col]
                [f"storage_{i}"], label=f"storage {i}", lw=0.25)

        ax.xaxis.set_major_formatter(fmt)
        ax.text(
            0.5, 0.87, f"Scénario {scenario} ({months[col]})", ha='center', transform=ax.transAxes)

        ax = axes_dispatch[row, col]
        for i in range(dp.shape[0]):
            if i == 0:
                base = 0
            else:
                base = np.sum([data[col][f"dispatch_{j}"] for j in np.arange(i)], axis=0)
            
            ax.fill_between(data[col].index.get_level_values(0), base, base+data[col]
                [f"dispatch_{i}"], label=f"dispatch {i}", alpha=0.5)

            ax.plot(data[col].index.get_level_values(0), base+data[col]
                [f"dispatch_{i}"], label=f"dispatch {i}", lw=0.25)

        ax.xaxis.set_major_formatter(fmt)
        ax.text(
            0.5, 0.87, f"Scénario {scenario} ({months[col]})", ha='center', transform=ax.transAxes)

    row += 1

for label in axes[-1, 0].get_xmajorticklabels() + axes[-1, 1].get_xmajorticklabels():
    label.set_rotation(30)
    label.set_horizontalalignment("right")

for label in axes_storage[-1, 0].get_xmajorticklabels() + axes_storage[-1, 1].get_xmajorticklabels():
    label.set_rotation(30)
    label.set_horizontalalignment("right")

for label in axes_dispatch[-1, 0].get_xmajorticklabels() + axes_dispatch[-1, 1].get_xmajorticklabels():
    label.set_rotation(30)
    label.set_horizontalalignment("right")

flex = "With" if args.flexibility else "Without"

plt.subplots_adjust(wspace=0, hspace=0)
fig.suptitle(f"Simulations based on {args.begin}--{args.end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)")
fig.text(1, 0, 'Lucas Gautheron', ha="right")
fig.legend(labels, loc='lower right', bbox_to_anchor=(1, -0.1),
           ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig("output.png", bbox_inches="tight", dpi=200)

fig_storage.suptitle(f"Simulations based on {args.begin}--{args.end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)")
fig_storage.text(1, 0, 'Lucas Gautheron', ha="right")
fig_storage.legend(labels_storage, loc='lower right', bbox_to_anchor=(1, -0.1),
           ncol=len(labels_storage), bbox_transform=fig_storage.transFigure)
fig_storage.savefig("output_storage.png", bbox_inches="tight", dpi=200)

fig_dispatch.suptitle(f"Simulations based on {args.begin}--{args.end} weather data.\n{flex} consumption flexibility; no nuclear seasonality (unrealistic)")
fig_dispatch.text(1, 0, 'Lucas Gautheron', ha="right")
fig_dispatch.legend(labels_dispatch, loc='lower right', bbox_to_anchor=(1, -0.1),
           ncol=len(labels_dispatch), bbox_transform=fig_dispatch.transFigure)
fig_dispatch.savefig("output_dispatch.png", bbox_inches="tight", dpi=200)
plt.show()
