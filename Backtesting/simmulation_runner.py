import os
import sys 
import json
import pandas as pd
import time 
import concurrent.futures
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from agents.quant_agent import run_quants_agent
from agents.signal_agent import run_signals_agent
from agents.decision_agent import run_decision_agent
from agents.reflect_2 import ReflectAgent
from agents.long_term_reflect import LongTermFeedbackEvaluator 

# ----------------------------
# Configurations
# ----------------------------
start_date = "2024-07-01"
end_date   = "2025-04-10"
state_dir  = os.path.join("state")
os.makedirs(state_dir, exist_ok=True)
# ----------------------------
# Utilities
# ----------------------------
def get_date_range(start, end):
    d1 = datetime.strptime(start, "%Y-%m-%d")
    d2 = datetime.strptime(end, "%Y-%m-%d")
    return [(d1 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((d2 - d1).days + 1)]

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_state_map():
    state_map = {}
    for fname in sorted(os.listdir(state_dir)):
        if fname.endswith(".json"):
            date = fname.replace(".json", "")
            with open(os.path.join(state_dir, fname), "r") as f:
                state_map[date] = json.load(f)
    return state_map


# ----------------------------
# Main Loop
# ----------------------------

def run_all_days(): 
    performance_log = {} # Store performance metrics for each day

    date_range = get_date_range(start_date, end_date)
    state_map = load_state_map()
    for date in sorted(state_map.keys()):
        state = state_map[date]
        if "performance_log" in state:
            performance_log[date] = state["performance_log"].get(date,None) # CHECK

    date_range = get_date_range(start_date, end_date) 

    for date in date_range:
        if date in state_map:
            print(f"⏩ Skipping {date} (already processed)")
            continue

        processed_dates = sorted(state_map.keys()) 
        prev_date = processed_dates[-1] if processed_dates else None
        prev_state = state_map.get(prev_date, {})
        prev_feedback = prev_state.get("reflect_output", {})
        prev_long_term_feedback = prev_state.get("long_term_feedback", {})   

        print(f"\n📅 Running agents for {date}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            long_term_weekly_quants = prev_long_term_feedback.get("weekly",{}).get("quants", "") 
            long_term_quants = f"Weekly: {long_term_weekly_quants}" 

            long_term_weekly_signals = prev_long_term_feedback.get("weekly",{}).get("signals", "") 
            long_term_signals = f"Weekly: {long_term_weekly_signals}" 

            future_quants = executor.submit(
                run_quants_agent,
                date = date,
                feedback = prev_feedback.get("quants", ""), 
                long_term=long_term_quants
                )
            
            future_signals = executor.submit(
                run_signals_agent,
                date = date, 
                feedback=prev_feedback.get("signals", ""),
                long_term=long_term_signals
                )

            quants_output,quants_private = future_quants.result() 
            signals_output,signals_private = future_signals.result() 


        # === DECISION ===
        long_term_weekly_decision = prev_long_term_feedback.get("weekly",{}).get("decision", "") 
        long_term_decision = f"Weekly: {long_term_weekly_decision}" 
        
        decision_output = run_decision_agent(
            quants_output, signals_output,
            feedback=prev_feedback.get("decision", ""),
            long_term=long_term_decision,
            risk_adjustment=None,
            curr_allocation = performance_log.get(prev_date,{}).get("decision_allocation",{}),
            curr_value = performance_log.get(prev_date,{}).get("portfolio_values",{}).get('decision',100),
        )

        # === REFLECT AGENT CLASS ===
        reflect = ReflectAgent(
            date=date,
            prev_date = prev_date,
            Quant_output=quants_output,
            Quant_portfolio=quants_private,
            Signal_output=signals_output,
            Signal_portfolio=signals_private,
            decision_output=decision_output,
            performance_log=performance_log,
        )

        feedback, updated_performance_log = reflect.run() 
        performance_log = updated_performance_log 
        
        # print(performance_log) 

        long_term = LongTermFeedbackEvaluator(
            performance_log=performance_log,
            date=date
        ) 
        # === LONG-TERM FEEDBACK AGENT ===
        long_term_feedback, long_term_updated_performance_log = long_term.run_long_term_feedback()
        performance_log = long_term_updated_performance_log

        # === SAVE STATE ===
        full_state = {
            "date": date,
            "quants_output": quants_output,
            "quants_private": quants_private,
            "signals_output": signals_output,
            "signals_private": signals_private,
            "decision_output": decision_output,
            "reflect_output": feedback,
            "long_term_feedback": long_term_feedback,
            "performance_log": {date: performance_log[date]},
        }

        with open(os.path.join(state_dir, f"{date}.json"), "w") as f:
            json.dump(full_state, f, indent=2)

        print(f"✅ Saved state for {date}")
        state_map[date] = full_state

# ----------------------------
# Execute
# ----------------------------
if __name__ == "__main__":
    start_time = time.time()  
    run_all_days()
    end_time = time.time()

    print("Simulation completed.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds") 

