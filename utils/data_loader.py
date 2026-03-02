import json
import pandas as pd
from collections import defaultdict
import os

# Load all big datasets

print("cwd=",os.getcwd())  
root_dir = os.path.join(os.getcwd(),"final_datasets") 
sentiment = json.load(open(os.path.join(root_dir,"btc_sentiment.json"))) 
fear_greed = json.load(open(os.path.join(root_dir,"crypto_fear_greed_index.json")))
news = json.load(open(os.path.join(root_dir,"daily_crypto_news_final_all.json"))) 

indicators_df = pd.read_csv(os.path.join(root_dir, "indicators_by_day.csv"))
onchain_df = pd.read_csv(os.path.join(root_dir, "BTC_on_chain_metrics.csv"))
ohlcv_df = pd.read_csv(os.path.join(root_dir, "BTCUSDT_daily_data_Binance.csv")) 

# Prep

sentiment_by_date = defaultdict(list)
for item in sentiment:
    item["date"] = pd.to_datetime(item["date"]).strftime("%Y-%m-%d")
    sentiment_by_date[item["date"]].append(item)

fg_by_date = {}
for entry in fear_greed:
    date = pd.to_datetime(entry["timestamp"]).strftime("%Y-%m-%d")
    fg_by_date[date] = {
        "value": int(entry["value"]),
        "classification": entry["value_classification"]
    }

news_by_date = defaultdict(list)
for item in news:
    date = pd.to_datetime(item["date"]).strftime("%Y-%m-%d")
    for article in item["articles"]:
        news_by_date[date].append(article)

indicators_df["date"] = pd.to_datetime(indicators_df["date"]).dt.strftime("%Y-%m-%d")
indicators_dict = indicators_df.set_index("date").to_dict(orient="index")

onchain_df["date"] = pd.to_datetime(onchain_df["date"]).dt.strftime("%Y-%m-%d")
onchain_dict = onchain_df.set_index("date").to_dict(orient="index")

ohlcv_df["date"] = pd.to_datetime(ohlcv_df["date"]).dt.strftime("%Y-%m-%d")
ohlcv_dict = ohlcv_df.set_index("date").to_dict(orient="index")

# Build daily files

# Get all unique dates across datasets
all_signal_dates = set(sentiment_by_date.keys()) | set(news_by_date.keys()) | set(fg_by_date.keys())
all_quant_dates = set(indicators_dict.keys()) | set(onchain_dict.keys())
all_dates = all_signal_dates | all_quant_dates 


for date in sorted(all_dates):
    signal_data = {
        "date": date,
        "sentiment": sentiment_by_date.get(date, []),
        "fear_greed": fg_by_date.get(date, {}),
        "news": news_by_date.get(date, [])
        }
    
    with open(f"data/signals/{date}.json", "w") as f:
        json.dump(signal_data, f, indent=2)

    
    quants_data = {
            "ohlcv":ohlcv_dict.get(date,{}), 
            "indicators": indicators_dict.get(date, {}),
            "onchain": onchain_dict.get(date, {})
        }
    with open(f"data/quants/{date}.json", "w") as f:
        json.dump(quants_data, f, indent=2)
    
    print(f"Processed date: {date}")
