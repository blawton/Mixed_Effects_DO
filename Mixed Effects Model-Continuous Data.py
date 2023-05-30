# +
# %matplotlib inline

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
# -

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

# # Starting with STS Continuous Data

sts=pd.read_csv("Data/STS_Continuous_Data.csv", index_col=0)


sts

#Formatting time
sts["ActivityStartDate"]=pd.to_datetime(sts["Date Time (GMT-04:00)"].str[0:10])
sts["ActivityStartTime/Time"]=sts["Date Time (GMT-04:00)"].str[10:]
sts

# # Starting with DO and Temp

model_data=sts.reset_index(drop=True).copy(deep=True)
model_data.rename(columns={"Temperature (C)": "temp", "Dissolved Oxygen (mg/L)":"do"}, inplace=True)

model_data.drop(["ActivityStartTime/Time"], axis=1, inplace=True)
print(len(model_data))
model_data.drop_duplicates(subset=["Station ID", "Date Time (GMT-04:00)"], inplace=True)
print(len(model_data))
model_data.set_index(["Embayment", "Station ID", "ActivityStartDate"], inplace=True)
model_data

print(len(model_data))
model_data.dropna(subset=["do", "temp"], inplace=True)
print(len(model_data))

#Getting counts of multi-parameter data by station
station_counts=model_data.reset_index().groupby("Station ID")["do"].count()
station_counts

#Getting counts of multi-parameter data by embayment
model_data.reset_index().groupby("Embayment")["do"].count()

model_data=model_data.reset_index().copy(deep=True)
model_data

#Getting year variable
model_data["year"]=model_data["ActivityStartDate"].dt.year

#Getting counts of multi-parameter data by year
model_data.groupby("year")["do"].count()

# ## Running the Actual Model

#Parameters
station_cutoff=10

#Ensuring indep/dep vars are numeric
model_data["do"]=pd.to_numeric(model_data["do"], errors="coerce")
model_data["temp"]=pd.to_numeric(model_data["temp"], errors="coerce")
model_data.dropna(subset=["do", "temp"], inplace=True)

#Dropping stations with two few observations (partly to avoid sing matrix)
station_drops=list(station_counts.loc[station_counts<station_cutoff].index)
processed=model_data.copy(deep=True)
processed.set_index("Station ID", inplace=True)
print(len(processed))
processed.drop(station_drops, axis=0, inplace=True)
print(len(processed))
processed.reset_index(inplace=True)
processed

#Running the actual model
data=processed
md=smf.mixedlm("do ~ temp + C(Embayment) + C(year)", data, groups=data["Station ID"])
mdf = md.fit(method=["Powell", "lbfgs"])
print(mdf.summary())

help(pd.DataFrame.drop)


