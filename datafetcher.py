import csv
import os
from binance import Client

client = Client(os.environ['API_KEY'], os.environ['API_SECRET'])

print("Fetching data...")
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 year ago UTC")

print("Writing to file...")
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i, candle in enumerate(klines):
        if not i % 1000:
            print(f"{i} / {len(klines)}")
        writer.writerow(candle)

print("Data saved")
