
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
def run_decision_agent(quants_output, signals_output, feedback="", long_term="", risk_adjustment=None , curr_allocation=None, curr_value=None):
    # Extract known values
    date = quants_output.get("date")  # assume both share the same date
    q_pred = quants_output.get("prediction", {})
    q_reason = quants_output.get("reasoning", {})

    s_pred = signals_output.get("prediction", {})
    s_reason = signals_output.get("reasoning", {})

    formatted_long_term_feedback = json.dumps(long_term,indent=2)

    # If risk agent is forcing constraints
    risk_note = ""
    if risk_adjustment and risk_adjustment.get("requires_adjustment", False):
        risk_note = risk_adjustment.get("reason", "")

    # Prompt construction
    system_prompt = f"""
You are an intelligent Bitcoin Portfolio Decision Strategist.

You will receive:
1. Technical + on-chain market analysis (from the Quants Agent)
2. Sentiment + news-based analysis (from the Signals Agent)
3. Optional risk adjustment feedback (from Risk Agent)
4. Current portfolio allocation and value
5. Performance feedback from Reflect Agent (short + long term) 
Short term feedback: {feedback.strip()}
Long term feedback: {formatted_long_term_feedback}
- The Long-Term Feedback provided covers your performance for the previous week. Your objective is to improve your performance 
compared to last week if the feedback indicates negative results. Focus on correcting the areas highlighted. Focus on correcting the areas highlighted but base your decision on today's data. Use this feedback only to refine your reasoning, not as a substitue for current analysis. Weight your current data analysis higher than this feedback provided.
- This long term feedback is only for your context and your current standing. 
- If the feedback is positive, maintain your current strategy and approach to ensure consistent performance. 
Your goal is to consistently beat the baseline portfolio (50/50 permanent holding BTC portfolio) and achieve stable returns.

Your task:
- Combine these insights to form a final prediction for the market (Bullish / Bearish / Neutral)
- Suggest a BTC vs Cash portfolio split (in %). You can increase, reduce or maintain the portfolio allocation based on the prediction.
- Justify your final allocation with reasoning based on the data provided
- Please ensure you analyze the feedback and adjust your reasoning accordingly.
- Provide detailed "key adjustments" analysis, clearly explaining the logic behind your allocation. 
  1. State the change in BTC allocation you are making and why? (e.g., "Increased BTC allocation from 20% → 60% to capitalize on bullish technical trend strength and news catalysts, while maintaining 40% cash to hedge overbought RSI risks.").
  2. Address the feedback and the long term feedback provided explicitly, indicating how it influenced your decision. (e.g., "Reflect feedback addressed by increasing exposure to avoid missed upside, but retained defensive buffer via cash allocation.").
  3. Balance Quant's technical insights with Signal's sentiment analysis, specifying how the two were combined. 
  These are example suggestions; do not use them word-for-word.
- Base your decision on today's data, not yesterday's feedback. Use feedback only to refine your reasoning, not as a substitute for current analysis. Weigh your current data analysis higher than the feedback provided.

IMPORTANT:
- Risk adjustments must be respected if present.
- Use the current portfolio allocation and value to make more context-aware decisions between increasing, decreasing, or holding the allocation.

ITS HIGHLY IMPORTANT YOU STRICTLY RESPOND IN THIS FORMAT ONLY, THE ENTIRE CODE BREAKS IF YOU DO NOT FOLLOW THIS:

{{
  "date": "YYYY-MM-DD",
  "final_prediction": {{
    "bullish": <int>,
    "bearish": <int>,
    "neutral": <int>
  }},
    "reasoning": {{
    "technical": "<string>",
    "sentiments": "<string>",
    "key_adjustments": "<string>"
  }},
  "final_allocation": {{
    "btc": <int>,
    "cash": <int>
  }}
}}
""".strip()

    # User message: give full context cleanly
    user_content = {
        "quants_prediction": q_pred,
        "quants_reasoning": q_reason,
        "signals_prediction": s_pred,
        "signals_reasoning": s_reason,
        "reflect_feedback_short": feedback,
        "reflect_feedback_long_term": long_term,
        "risk_feedback": risk_note,
        "current_allocation": curr_allocation,
        "current_value": curr_value
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Today's market context for {date}:\n\n{json.dumps(user_content, indent=2)}\n\nREMINDER: You must respond strictly in the output format provided. Do not add extra text or headings."}
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

    # Parse response JSON
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        print(f"❌ JSON parse error on decision agent {date}")
        print(f"Response content: {content}")
        raise

    result["date"] = date

    output_dict = {
        "date": result["date"],
        "final_prediction": result["final_prediction"],
        "reasoning": result["reasoning"],
        "final_allocation": result["final_allocation"]
    }

    print(f"[✓] Decision Agent finished for {date}")
    return output_dict

""" 
**Key Adjustments:**
1. Increased BTC allocation from 20% → 60% to capitalize on bullish technical trend strength and structural news catalysts, while maintaining 40% cash to hedge overbought RSI risks.
2. Reflect feedback addressed by increasing exposure to avoid missed upside, but retained defensive buffer via cash allocation.
3. Balanced Quants' high-conviction technicals (60% bullish) with Signals' moderate bullishness (45%) through weighted averaging.
4. Maintained risk-awareness via cash reserve despite absence of explicit Risk Agent constraints.

"""