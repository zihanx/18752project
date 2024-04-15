import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def find_startline(filename, string_to_search):
    with open(filename, 'r') as read_obj:
        # Read all lines in the file one by one
        for i, line in enumerate(read_obj):
            # For each line, check if line contains the string
            if string_to_search in line:
                return i
        return False
    
year_span = 10
year_start = 2010
date_low = int(str(year_start) + "0101")  #20050101
date_high = int(str(year_start + year_span) + "0101") #20150101
print(f"Collect from {date_low} until {date_high}")

data_path = "data\\ECAD_DATA\\DUSSELDORF"
data_collection = {}


feature_types = {"TG_": "temp_mean",
                 "TN_": "temp_min",
                 "TX_": "temp_max",
                 "SS_": "sunshine",
                 "CC_": "cloud_cover",
                 "FG_": "wind_speed",
                 "FX_": "wind_gust",
                 "HU_": "humidity",
                 "PP_": "pressure",
                 "QQ_": "global_radiation",
                 "RR_": "precipitation",
                 "SS_": "sunshine"}

#data_collection[folder] = {}
for file in os.listdir(data_path):
    if file[:3] in feature_types.keys():
        header_line = find_startline(os.path.join(data_path, file), "SOUID,")
        if header_line:
            df = pd.read_csv(os.path.join(data_path, file), header=header_line,
                            skip_blank_lines=False)
            df.columns = [x.strip() for x in df.columns]
            df_select = df[(df["DATE"] >= date_low) & (df["DATE"] <= date_high)]
            # Add column values to collection
            # Add date
            data_collection["DATE"] = df_select["DATE"].values
            data_collection["MONTH"] = [int(str(x)[4:6]) for x in df_select["DATE"].values]
            data_collection[feature_types[file[:3]]] = df_select[file[:2]].values
            
dataset = pd.DataFrame.from_dict(data_collection)

# clean up the data
# if invalid fraction is greater than 5%, drop it
drop_columns = []
for column in dataset.columns:
    invalid_values = np.sum(dataset[column].values == -9999)
    percentage_invalid = 100 * invalid_values/(year_span*365.25)
    if percentage_invalid > 1:
        print(column, f"invalid fraction: {percentage_invalid:.3f}")    
    if percentage_invalid > 5:
        print("--> drop column,", column)
        drop_columns.append(column)
dataset = dataset.drop(columns=drop_columns)

# replace -9999 value with the mean of all other valid values
for column in dataset.columns:
    idx = np.where(dataset[column].values == -9999)[0]
    if idx.shape[0] > 0:
        mean_value = dataset[column][dataset[column] != -9999].mean()
        print(f"Replace {idx.shape[0]} -9999 values in {column} by mean of {mean_value}")
        dataset[column].values[idx] = dataset[column][dataset[column] != -9999].mean()

# Normalize the data to a smaller range
for column in dataset.columns:
    if "humidity" in column:
        dataset[column] = dataset[column] / 100
    elif "pressure" in column:
        dataset[column] = dataset[column] / 10000
    elif "temp" in column:
        dataset[column] = dataset[column] / 10
    elif "sunshine" in column:
        dataset[column] = dataset[column] / 10
    elif "wind_speed" in column:
        dataset[column] = dataset[column] / 10
    elif "wind_gust" in column:
        dataset[column] = dataset[column] / 10
    elif "global_radiation" in column:
        dataset[column] = dataset[column] / 100
    elif "precipitation" in column:
        dataset[column] = dataset[column] / 100

dataset.to_csv("data\\preprocessed_weather_dataset.csv", index=False)


# Create sleep day label for the dataset
sleep_weather = {}
sleep_weather["DATE"] = dataset["DATE"].values


cloud_cover = None
temp_max = None
for x in dataset.columns:
    if "cloud_cover" in x:
        cloud_cover = dataset[x].values
    elif "temp_max" in x:
        temp_max = dataset[x].values
        
    if cloud_cover is not None and temp_max is not None:
        sleep_weather["sleep_weather"] = ((cloud_cover > 5) & (temp_max <= 20))
        break

labels_sleep = pd.DataFrame.from_dict(sleep_weather)
labels_sleep.to_csv("data\\weather_prediction_sleep_labels.csv", index=False)

key = "sleep_weather"
plt.hist(sleep_weather[key].astype(int))
plt.title(key)
plt.show()

# For the real dataset, drop cloud_cover and temp_max because we
# used them for label generation
dataset.drop(["cloud_cover", "temp_max"], axis=1, inplace=True)

# Manual feature engineering:
# Create a comfort index integrating temp_mean and humidity
def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

# assume the comfort score follows gaussian distribution, closer to optimal point, larger the score.
def comfort_score(temp_mean, humidity, optimal_temp, temp_sigma, optimal_humidity, humidity_sigma):
    temp_score = gaussian(temp_mean, optimal_temp, temp_sigma)
    humidity_score = gaussian(humidity, optimal_humidity, humidity_sigma)
    # Combine the temperature and humidity scores
    combined_score = (temp_score + humidity_score) / 2
    return combined_score

optimal_temp = 18
optimal_humidity = 0.6
temp_sigma = dataset['temp_mean'].std()
humidity_sigma = dataset['humidity'].std()
dataset['comfort_score'] = dataset.apply(lambda row: comfort_score(row['temp_mean'], row['humidity'], optimal_temp, temp_sigma, optimal_humidity, humidity_sigma), axis=1)
# dataset = dataset.merge(labels_sleep, on='DATE', how='left')
dataset.to_csv("data\\weather_dataset.csv", index=False)