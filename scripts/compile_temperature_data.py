from re import L
import pandas as pd
import numpy as np
import netCDF4 as nc
import os

import datetime

from matplotlib import pyplot as plt

years = np.arange(1980, 2022 + 1)

data = []

for year in years:
    dir = f"data/urs/{year}"

    if os.path.exists(dir):
        files = os.listdir(dir)
    else:
        continue

    for f in files:
        try:
            ds = nc.Dataset(os.path.join(dir, f))
        except:
            continue

        metadata = ds.__dict__
        day = datetime.datetime.strptime(metadata["RangeBeginningDate"], "%Y-%m-%d")

        print(day)

        times = np.array(ds.variables["time"])
        lat = np.array(ds.variables["lat"])
        lon = np.array(ds.variables["lon"])

        n_times = len(times)
        n_lat = len(lat)
        n_lon = len(lon)

        times = np.tile(np.reshape(times, (n_times, 1, 1)), [1, n_lat, n_lon])
        lat = np.tile(np.reshape(lat, (1, n_lat, 1)), [n_times, 1, n_lon])
        lon = np.tile(np.reshape(lon, (1, 1, n_lon)), [n_times, n_lat, 1])

        df = {
            "times": times,
            "lat": lat,
            "lon": lon,
            "T2M": np.array(ds.variables["T2M"]),
            "T10M": np.array(ds.variables["T10M"]),
        }

        for k in df:
            df[k] = df[k].flatten()

        df = pd.DataFrame(df)
        df["times"] = day + pd.TimedeltaIndex(df["times"], unit="minutes")

        data.append(df)

data = pd.concat(data)
data.sort_values(["times", "lat", "lon"], inplace=True)
data.to_parquet("data/temperatures.parquet")
