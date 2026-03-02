
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
def run_signals_agent(date, feedback="", long_term=""):
    print("Starting signals")
    # Load the sentiment + news data for the given date
    data_path = os.path.join("data", "signals", f"{date}.json")
    with open(data_path, "r") as f:
        file_data = json.load(f)
    
    formatted_long_term_feedback = json.dumps(long_term, indent = 2)

    # Prepare system prompt with both feedback types
    system_prompt = f"""
You are a Market Sentiment Analyst specializing in analyzing Bitcoin-related sentiment data to predict the next day’s market stance using:
1. Social sentiment scores 
2. Fear and Greed Index
3. Recent crypto-relevant news headlines

Recent feedback: {feedback.strip()}
Long-term feedback: {formatted_long_term_feedback}
- The Long-Term Feedback provided covers your performance for the previous week. Your objective is to improve your performance 
compared to last week if the feedback indicates negative results. Focus on correcting the areas highlighted.Focus on correcting the areas highlighted but base your decision on today's data. Use this feedback only to refine your reasoning, not as a substitue for current analysis. Weight your current data analysis higher than this feedback provided.
- This long term feedback is only for your context and your current standing. 
- If the feedback is positive, maintain your current strategy and approach to ensure consistent performance. 
- Base your decision on today's data, not yesterday's feedback. Use feedback only to refine your reasoning, not as a substitute for current analysis. Weigh your current data analysis higher than the feedback provided.
Your goal is to consistently beat the baseline portfolio (50/50 permanent holding BTC portfolio) and achieve stable returns.

Your task:
- Output prediction probabilities for Bullish, Bearish, and Neutral market.
- Provide your reasoning for sentiment indicators and news signals separately.
- Suggest a portfolio split between BTC and Cash based on the prediction.

ITS HIGHLY IMPORTANT YOU Respond strictly in this Format only:

{{
  "date": "YYYY-MM-DD",
  "prediction": {{
    "bullish": <int>,
    "bearish": <int>,
    "neutral": <int>
  }},
  "reasoning": {{
    "sentiment": "<string>",
    "news": "<string>"
  }},
  "suggested_portfolio": {{
    "btc": <int>,
    "cash": <int>
  }}
}}

It is highly important to:
- Use the sentiment scores (score1, score2, score3, mean, sum) and count to assess overall market mood.
- Interpret the Fear and Greed Index to understand market sentiment.
- Evaluate the tone and implications of each news article to identify overall market.
= Base your decision on today's data, not yesterday's feedback. Use feedback only to refine your reasoning, not as a substitute for current analysis. Weigh your current data analysis higher than the feedback provided.
- Justify how each signal leads to your prediction clearly and concisely.
- Do not mention any additional information apart from the instructions in the output format. 
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the sentiment and news data for {date}:\n\n{file_data}\n\nPredict the market for tomorrow based on this data.\n REMINDER: You must respond strictly in the output format provided. Do not add extra text or headings."}
    ]

    # Call DeepSeek Reasoner
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )

    content = response.choices[0].message.content

    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    
    start = content.find('{')
    end = content.rfind('}') + 1
    content = content[start:end]

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

    print(f"[✓] Signals Agent finished for {date}")
    return output_dict, private_dict
