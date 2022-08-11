wind_data = pd.read_csv("ninja_wind_europe_v1.1_current_on-offshore.csv")
solar_data = pd.read_csv("ninja_pv_europe_v1.1_merra2.csv")

wind_data['time'] = pd.to_datetime(wind_data['time'], format="%Y-%m-%d %H:%M:%S")
solar_data['time'] = pd.to_datetime(solar_data['time'], format="%Y-%m-%d %H:%M:%S")

regions_labels = sorted(list(solar_data.columns))
regions_labels.remove("time")

offshore_labels = {x for x in wind_data.columns if '_OFF' in x}
missing_offshore = {f"{x}_OFF" for x in regions_labels} - offshore_labels

for x in missing_offshore:
    wind_data[x] = 0

offshore_labels = [x for x in wind_data.columns if '_OFF' in x]
onshore_labels = [x for x in wind_data.columns if '_ON' in x]

offshore = wind_data[["time"] + offshore_labels]
onshore = wind_data[["time"] + onshore_labels]
solar = solar_data.set_index("time").stack().rename("solar")
print(solar.index.names)
solar.index.names = ["time", "region"]

offshore.set_index("time", inplace=True)
offshore.columns = map(lambda x: x.replace("_OFF", ""), offshore.columns)
offshore = offshore.stack().rename("offshore")
offshore.index.names = ["time", "region"]

onshore.set_index("time", inplace=True)
onshore.columns = map(lambda x: x.replace("_ON", ""), onshore.columns)
onshore = onshore.stack().rename("onshore")
onshore.index.names = ["time", "region"]

potential = pd.DataFrame(offshore)
print(potential)
potential = potential.merge(pd.DataFrame(onshore), left_index=True, right_index=True, how="outer")
print(potential)
potential = potential.merge(pd.DataFrame(solar), left_index=True, right_index=True, how="outer")
print(potential)
potential.sort_values(["time", "region"], inplace=True)

potential.to_parquet("potential.parquet")