import csv
import os
from binance import Client

client = Client(os.environ['API_KEY'], os.environ['API_SECRET'])

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 month ago UTC")

with open('data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for candle in klines:
        spamwriter.writerow(candle)

print("Data saved")
