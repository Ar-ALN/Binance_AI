import csv
from binance import Client

client = Client('TniMYW80Nz5VRK9v0xkIHMTcIKajsFXBzkNZLKEgsuANdITibck2NwKCfGEsE7sG',
                'cHnla50KhvEagVRcvyGgvxs541GrNFfjDdgrm3lN1k7hRocniDk391VzlLeE6P3D')

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 month ago UTC")

with open('data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for candle in klines:
        spamwriter.writerow(candle)

print("Data saved")
