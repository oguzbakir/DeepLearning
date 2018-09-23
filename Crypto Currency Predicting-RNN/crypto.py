import pandas as pd

SEQ_LEN = 60  # Minutes of data
FUTURE_PERIOD_PREDICT = 3  # Minutes of prediction
RATIO_TO_PREDICT = "LTC-USD"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
    dataset = "crypto_data/{}.csv".format(ratio)

    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": "{}_close".format(ratio), "volume": "{}_volume".format(ratio)}, inplace=True)

    df.set_index("time", inplace=True)
    df = df[["{}_close".format(ratio), "{}_volume".format(ratio)]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df["{}_close".format(RATIO_TO_PREDICT)].shift(-FUTURE_PERIOD_PREDICT)  # Set future column to be shifted value


main_df['target'] = list(map(classify, main_df["{}_close".format(RATIO_TO_PREDICT)], main_df['future']))

print(main_df[["{}_close".format(RATIO_TO_PREDICT), "future", "target"]].head(10))
