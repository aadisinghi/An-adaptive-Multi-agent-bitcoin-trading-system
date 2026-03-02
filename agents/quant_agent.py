""" 
def run_quants_agent(date, feedback):
    # Loads internally from data/quants/{date}.json
    # Uses feedback in prompt
    ...
    return output_dict, private_dict



run_quants_agent(date, feedback="", long_term="")
``And expects a return of:
```python
return output_dict, private_dict
``Where:
- `output_dict` includes `prediction`, `reasoning`, and `date`
- `private_dict` includes `suggested_portfolio` (not visible to Decision)

You already use DeepSeek via `client.chat.completions.create(...)`.  
Now we just make it modular, clean, and plug-and-play.

---

## ✅ Final `quants_agent.py` (cleaned + compatible)

```python
"""  

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load DeepSeek API credentials
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# === Main Function Called by Runner ===
def run_quants_agent(date, feedback="", long_term=""):
    print("Starting quants")
    # Load the data for the given date
    data_path = os.path.join("data", "quants", f"{date}.json")
    with open(data_path, "r") as f:
        file_data = json.load(f)
    
    formatted_long_term_feedback = json.dumps(long_term, indent=2)

    # Prepare system prompt with feedback
    system_prompt = f"""
You are a Quantitative Trading Analyst analyzing Bitcoin market data to predict the next day’s market stance using:
1. OHLCV data
2. Technical indicators
3. On-chain metrics

Recent feedback: {feedback.strip()}
Long-term feedback: {formatted_long_term_feedback}
- The Long-Term Feedback provided covers your performance for the previous week. Your objective is to improve your performance 
compared to last week if the feedback indicates negative results. Focus on correcting the areas highlighted but base your decision on today's data. Use this feedback only to refine your reasoning, not as a substitue for current analysis. Weight your current data analysis higher than this feedback provided.
- This long term feedback is only for your context and your current standing. 
- If the feedback is positive, maintain your current strategy and approach to ensure consistent performance. 
Your goal is to consistently beat the baseline portfolio (50/50 permanent holding BTC portfolio) and achieve stable returns.
- Base your decision on today's data, not yesterday's feedback. Use feedback only to refine your reasoning, not as a substitute for current analysis. Weigh your current data analysis higher than the feedback provided.

Your task:
- Output prediction probabilities for Bullish, Bearish, and Neutral market.
- Provide your reasoning for technicals and on-chain metrics separately.
- Suggest a portfolio split between BTC and Cash based on the prediction.

ITS HIGHLY IMPORTANT YOU Respond strictly in this Format ONLY:

{{
  "date": "YYYY-MM-DD",
  "prediction": {{
    "bullish": <int>,
    "bearish": <int>,
    "neutral": <int>
  }},
  "reasoning": {{
    "technical": "<string>",
    "on_chain": "<string>"
  }},
  "suggested_portfolio": {{
    "btc": <int>,
    "cash": <int>
  }}
}}

It's highly important to provide an accurate reasoning for your predictions by going through each of the provided data points
and giving a clear explanation of how they lead to your conclusion. You have to go through each of the provided facts and reason your prediction: MACD, RSI, Bollinger bands, adx, vwap,adx, taker buy ratio, ohlcv data, on chain 3 metrics.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the market data for {date}:\n\n{file_data}\n\nPredict the market for tomorrow based on this data."}
    ]

    # Call DeepSeek Reasoner
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )

    content = response.choices[0].message.content

    # Parse JSON safely
    try:
        result_json = json.loads(content)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON from model response on {date}")
        print(f"Response content: {content}")
        raise

    result_json["date"] = date

    # Split into public and private outputs
    output_dict = {
        "prediction": result_json["prediction"],
        "reasoning": result_json["reasoning"],
        "date": result_json["date"]
    }

    private_dict = {
        "suggested_portfolio": result_json["suggested_portfolio"]
    }

    print(f"[✓] Quants Agent finished for {date}")
    return output_dict, private_dict
