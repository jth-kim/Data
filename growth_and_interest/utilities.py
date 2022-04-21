import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint, acf, pacf
from arch import arch_model

def grab_data(labels):
    """
    Read data from my personal directory
    """
    directory = "/System/Volumes/Data/Users/jonathankim/Documents/hieros_gamos/\
treachery/kepler/professional_pf/statistical_and_data/data/rates_growth_data/cleaned_data/"
    dfs = {}
    for d in labels:
        pwd = directory + d + "/{}_data.csv".format(d)
        df = pd.read_csv(pwd)
        df.set_index("date", inplace=True)
        dfs[d] = df
    return dfs

def append_yoy_dgdp(data, labels, transformation="yoyg"):
    """
    We are analysing GDP growth. So yoyg that nominal GDP
    """
    for d in labels:
        df = data[d]
        target = data[d].gdp
        if transformation == "yoyg":
            df["dgdp"] = (target.shift(-1) / target) -1
        else:
            return

def produce_correlations_with_lags(data, labels, lags=4):
    """
    Produce linear correlation results
    """
    lag_range = range(-lags, lags+1)
    corr_df = pd.DataFrame(index=lag_range, columns=labels)
    for d in labels:
        df = data[d].copy().dropna()
        for lag in lag_range:
            df["rate_lag={}".format(lag)] = df.rate.shift(lag)
        corrs = df.corr()
        for lag in lag_range:
            corr_df.loc[lag, d] = corrs.loc["dgdp", "rate_lag={}".format(lag)]
    return corr_df, df

def series_stationarity_test(series, label, verbose=False, p_significance=0.05):
    """
    Test for stationarity
    """
    dftest = adfuller(series, autolag="AIC")
    out = pd.Series(dftest[0:4], index=["Statistic", "p_val", "nLags", "nObs"])
    result = None
    if out.p_val < 0.05:
        result = "H0 rejected".format(label)
    else:
        result = "h0 not rejected".format(label)
    if verbose == True:
        print("\nADF Results for {}:".format(label))
        if out.p_val < 0.05:
            print("PLAUSIBLY REJECTED ❌\\n")

        else:
            print("No rejection 🙅‍♀️\n")
        display(out[["Statistic", "p_val"]])
        print(dftest[4])
    return out, result

def series_cointegrated_test(df, verbose=False, p_significance=0.05):
    """
    Test for cointegration between pairs
    """
    df=df.dropna()
    cointegration_test = coint(df.gdp, df.rate)
    out = pd.Series(cointegration_test, index=["Statistic", "p_val", "crit"])
    result = "H0 rejected" if out.p_val < p_significance else "H0 not rejected"
    if verbose:
        if out.p_val < p_significance:
            print("H0: 'Not cointegrated' rejected")
        else:
            print("Cannot reject H1")
    return out, result


def display_test_results(data, labels):
    """
    pretty display for cointegration and stationarity tests
    """
    flattened_data = {}
    ccols = ["cointegrated", "coint statistic", "coint 1% sig", "coint pval"]
    coint_results = pd.DataFrame(index=labels, columns=ccols)

    for name, df in data.items():
        df = df.dropna()
        # Save for stationarity tests
        flattened_data["{}-dgdp".format(name)] = df.dgdp
        flattened_data["{}-rate".format(name)] = df.rate
        ctest, cresult = series_cointegrated_test(df, p_significance=0.25)
        coint_results.loc[name] = [cresult, ctest[0], ctest[2][0], ctest[1]]

    scols = ["stationary", "stat statistic", "stat pval"]
    stationarity_results = pd.DataFrame(index=flattened_data.keys(), columns=scols)

    for name, series in flattened_data.items():
        stest, sresult = series_stationarity_test(series, name)
        stationarity_results.loc[name] = [sresult, stest[0], stest[1]]

    display(coint_results.drop(["coint 1% sig"], axis=1))
    display(stationarity_results.drop(["stationary"], axis=1))
    return stationarity_results

def difference_non_cointegrated(data, stationarity, targets, method="percent"):
    """
    Difference a set of target data
    """
    output = data
    for i in targets:
        for j, row in stationarity.iterrows():
            if i in j and row["stat pval"] > 0.25:
                target_label, target_series_id = j.split("-")
                target_series = output[target_label][target_series_id].dropna()
                # - Absolute differencing
                if method == "abs":
                    updated_target = target_series - target_series.shift(-1)
                # - percentage calculation
                elif method == "percent":
                    updated_target  = 100 * ((target_series.shift(-1) / target_series) - 1)
                output[target_label][target_series_id] = updated_target
    return output