import csv
import os
from binance import Client

client = Client(os.environ['API_KEY'], os.environ['API_SECRET'])

print("Fetching data...")
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "2 year ago UTC")

print("Writing to file...")
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",

        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore"
    ])
    for i, candle in enumerate(klines):
        if not i % 1000:
            print(f"{i} / {len(klines)}")
        writer.writerow(candle)

print("Data saved")
