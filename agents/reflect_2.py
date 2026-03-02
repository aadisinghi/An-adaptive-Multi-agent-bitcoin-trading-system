import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd  
from datetime import datetime, timedelta
import glob 
import copy 

class ReflectAgent:

    def __init__(self, date,prev_date, Quant_output, Quant_portfolio, Signal_output, Signal_portfolio, decision_output,performance_log):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com") 

        self.date = date 
        self.prev_date = prev_date
        self.quant_output = Quant_output
        self.quant_portfolio = Quant_portfolio
        self.signal_output = Signal_output
        self.signal_portfolio = Signal_portfolio
        self.decision_output = decision_output
        self.performance_log = performance_log  # <- we have to update each iteration
        default_portfolio = {"quants": 100, "signals": 100, "decision": 100}
        self.portfolio_values = self.performance_log.get(self.prev_date, {}).get("portfolio_values", default_portfolio) # {"quants":100, "signals":100, "decision":100} 
        self.old_portfolio_values = copy.deepcopy(self.portfolio_values)

    def load_eval_data(self): 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path_for_daily_btc = os.path.join(base_dir, "..", "final_datasets", "btc_daily_returns_for_eval.csv")
        path_for_baseline = os.path.join(base_dir,"..","final_datasets","baseline_performance_full.csv")

        # Load the daily data and baseline data for the given date

        daily_data = pd.read_csv(path_for_daily_btc, index_col="date", parse_dates=True)
        baseline_data = pd.read_csv(path_for_baseline, index_col="date", parse_dates=True)

        # Filter the data for the specific date
        self.daily_data = daily_data.loc[self.date]
        self.baseline_data = baseline_data.loc[self.date]


    def evaluate_accuracy(self):
        """
        Returns soft accuracy scores between 0 and 1 for each agent.
        1.0 means the agent predicted the correct class as top prediction.
        A fractional score means the correct class was predicted with less confidence than the top one.
        """

        # output as follows: 
        #     {
        #   "quants": 0.5213
        #   "signals": 1.000
        #   "decision": 0.8443
        # }

        btc_return = self.daily_data["btc_return_daily"]
        real_result = "bullish" if btc_return > 0.5 else "bearish" if btc_return < -0.5 else "neutral" 

        def soft_accuracy(prediction_dict):
            pred_score_of_real_result = prediction_dict.get(real_result, 0)
            top_score = max(prediction_dict.values())
            if top_score == 0:
                return 0
            return round(pred_score_of_real_result / top_score, 4)

        return {
        "quants": soft_accuracy(self.quant_output["prediction"]),
        "signals": soft_accuracy(self.signal_output["prediction"]),
        "decision": soft_accuracy(self.decision_output["final_prediction"])
        }
    
    def get_allocation(self, agent):
        if agent == "quants":
            return self.quant_portfolio["suggested_portfolio"]
        elif agent == "signals":
            return self.signal_portfolio["suggested_portfolio"]
        elif agent == "decision":
            return self.decision_output["final_allocation"]
    
    def compute_daily_returns(self):
        btc_return = self.daily_data["btc_return_daily"]

        def get_return(allocation):
            btc_weight = allocation.get("btc", 0)/100
            return btc_return * btc_weight / 100

        return {
        agent: get_return(self.get_allocation(agent))
        for agent in ["quants", "signals", "decision"]
    }

    def compute_portfolio_value(self):
        # update self.portfolio_values based on daily returns. 
        daily_returns = self.compute_daily_returns()

        for agent in self.portfolio_values.keys():
            portfolio_return = daily_returns[agent]
            self.portfolio_values[agent] *= (1 + portfolio_return)
    
        return self.portfolio_values

    def compute_daily_portfolio_returns(self): 
        returns = {} 
        initial_portfolio_value = self.old_portfolio_values

        for agent in self.portfolio_values.keys():
            initial_portfolio_value_agent = initial_portfolio_value[agent]
            new_portfolio_value_agent = self.portfolio_values[agent]
            returns[agent] = round(((new_portfolio_value_agent - initial_portfolio_value_agent)/initial_portfolio_value_agent)*100,2) 

        return returns 
    
    def compute_regret(self):
        # compute regret as the difference between the portfolio value and baseline value
        baseline_value_today = self.baseline_data["portfolio_value"]
        portfolio_values = self.portfolio_values

        regret = {agent: value - baseline_value_today for agent, value in portfolio_values.items()}
        return regret
    
    def compute_cumulative_return_since_day_one(self):
        # compute cumulative return as the difference between the portfolio value and the initial value
        initial_value = 100
        portfolio_values = self.portfolio_values
        cumulative_return = {agent: (value - initial_value) / initial_value * 100 for agent, value in portfolio_values.items()}
        return cumulative_return

    def compute_sharpe_ratio(self,returns):
        # should recieve input as a list of returns. 
        # compute sharpe ratio as the mean return divided by the std of returns 
        if len(returns) == 0:
            return np.nan 
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0
        
        return round(mean_return / std_return , 4)
    
    def compute_weekly_monthly_returns(self,returns):
        if len(returns) == 0:
            return np.nan 
        
        mean_return = round(np.mean(returns),4)
        return mean_return 


    def build_prompt(self, quants_reasoning, signals_reasoning, decision_output, btc_return, returns, regret, accuracy, current_portfolio):
        system_prompt = """
        You are a performance evaluation agent. You will be given today's market data, predictions, reasoning, and outcomes from three agents: quants, signals, and decision.
        Context: Quants agent analyzes technical and on chain data and also suggests a portfolio, Signals agent analyzes sentiment and news data and suggest a portfolio, and Decision agent combines these insights to make a final prediction and portfolio allocation.
        Your task:
        - Assess whether each agent made justified decisions based on their reasoning and actual outcomes.
        - Identify what went right or wrong.
        - Return actionable feedback per agent.
        - Your feedback is the current days evaluation that is provided to each agent for their next trading session.
        - It is highly important to note: Feedback must be grounded in each agent's reasoning quality and prediction accuracy, not just outcome-based metrics like regret or returns.
        - It is also important that the feedback suggests how to improve the reasoning process for future predictions.
        - Understand that we are trying to optimize the profitability of the portfolio in the long term.
        - For the quants and signals agent. make sure you include suggestions pertaining to their reasonings along with portfolio profitability, and not only focus on the portfolio part because their reasoning is provided to the decision agent so its important they reason well with the provided inputs.

        Note: You are given metrics like 'accuracy', 'regret', and 'returns' per agent. Use these to judge each agent individually:
        - Accuracy shows whether the agent's predicted market direction matched the actual outcome, with higher values indicating closer alignment.(0-1) where 1 is best and most accurate.
        - Regret shows the difference between the portfolio values compared to baseline portfolio value (50/50 allocation) where positive implies higher return and negative implies lower.
        - Use the agent's own reasoning and these results to provide personalized feedback.
        And should also take into account the decision agent's (main agent) current portfolio value provided and the returns generated. THIS IS IMPORTANT BECAUSE WE ARE OPTIMIZING FOR IT.
        - Do not give explicit portfolio suggestions to any agent, it biases their real reasoning. 
        Important instructions carefully to be followed: 
        - Each agent is independently responsible for its own portfolio value and operates without visibility into the other agents' inputs. The Quants and Signals agents work in isolation, while only the Decision Agent has access to both their outputs.
        - Your feedback must focus solely on improving the agents reasoning using the existing inputs they already receive. Do not suggest actions or improvements that require additional data or external information beyond what they currently recieve. 
        - IMPORTANT: The Signals Agent does not have access to technical indicators or market data. Do not suggest that it should have considered technical trends. Feedback must be based solely on sentiment and news reasoning for Signals Agent.
        Its highly important you response strictly in the provided format, DO NOT include markdown syntax (e.g., \``json). Only return the raw JSON.`:

        {
        "feedback": {
            "quants": "<string>",
            "signals": "<string>",
            "decision": "<string>"
            }
        }
        """.strip()


        user_input = {
            "btc_return": btc_return,
            "agent_returns": returns,
            "regret": regret,
            "accuracy": accuracy,
            "quants_reasoning": quants_reasoning,
            "signals_reasoning": signals_reasoning,
            "decision_output": decision_output,
            "current_portfolio": current_portfolio,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_input, indent=2)}
        ]

        response = self.client.chat.completions.create(
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

        try:
            result = json.loads(content) 
        except json.JSONDecodeError:
            print(f"❌ JSON parse error on feedback agent {self.date}")
            print(f"Response content: {content}")
            raise 

        feedback_dict = {
            "quants": result["feedback"].get("quants", ""),
            "signals": result["feedback"].get("signals", ""),
            "decision": result["feedback"].get("decision", "")
        }

        print(f"[✓] Reflect Agent finished for {self.date}")

        return feedback_dict 
    
    def calculate_sharpe_ratios(self):
        sorted_dates = sorted(self.performance_log.keys())

        # Determine the start of the week and month based on non-overlapping periods
        completed_weeks = []
        completed_months = []

        # Build non-overlapping weeks
        for i in range(0, len(sorted_dates), 7):
            week = sorted_dates[i:i+7]
            if len(week) == 7:
                completed_weeks.append(week)

        # Build non-overlapping months (30-day periods)
        for i in range(0, len(sorted_dates), 30):
            month = sorted_dates[i:i+30]
            if len(month) == 30:
                completed_months.append(month)

        def get_agent_returns(agent, date_window):
            returns = []
            for date in date_window:
                if date in self.performance_log and "daily_returns" in self.performance_log[date]:
                    returns.append(self.performance_log[date]["daily_returns"][agent]) 
            return returns 
        
        sharpe_ratios = {agent: {"weekly": None, "monthly": None} for agent in ['quants', 'signals', 'decision']}

        # Calculate Sharpe Ratios for the last completed week only (7-day block)
        if completed_weeks:
            last_week = completed_weeks[-1]  # Last complete week
            if self.date == last_week[-1]:   # Only calculate on the last day of the week
                for agent in ['quants', 'signals', 'decision']:
                    weekly_returns = get_agent_returns(agent, last_week)
                    # print(f"{agent}:{weekly_returns}")      
                    sharpe_ratios[agent]["weekly"] = self.compute_sharpe_ratio(weekly_returns)
                    # print(sharpe_ratios) 

        # Calculate Sharpe Ratios for the last completed month only (30-day block)
        if completed_months:
            last_month = completed_months[-1]  # Last complete month
            if self.date == last_month[-1]:    # Only calculate on the last day of the month
                for agent in ['quants', 'signals', 'decision']:
                    monthly_returns = get_agent_returns(agent, last_month)
                    sharpe_ratios[agent]["monthly"] = self.compute_sharpe_ratio(monthly_returns)

        return sharpe_ratios
    

    def calculate_weekly_and_monthly_returns(self):
        sorted_dates = sorted(self.performance_log.keys())

        # Determine the start of the week and month based on non-overlapping periods
        completed_weeks = []
        completed_months = []

        # Build non-overlapping weeks
        for i in range(0, len(sorted_dates), 7):
            week = sorted_dates[i:i+7]
            if len(week) == 7:
                completed_weeks.append(week)

        # Build non-overlapping months (30-day periods)
        for i in range(0, len(sorted_dates), 30):
            month = sorted_dates[i:i+30]
            if len(month) == 30:
                completed_months.append(month)

        def get_agent_returns(agent, date_window):
            returns = []
            for date in date_window:
                if date in self.performance_log and "daily_returns" in self.performance_log[date]:
                    returns.append(self.performance_log[date]["daily_returns"][agent]) 
            return returns 
        
        weekly_monthly_returns = {agent: {"weekly": None, "monthly": None} for agent in ['quants', 'signals', 'decision']}

        if completed_weeks:
            last_week = completed_weeks[-1]  # Last complete week
            if self.date == last_week[-1]:   # Only calculate on the last day of the week
                for agent in ['quants', 'signals', 'decision']:
                    weekly_returns = get_agent_returns(agent, last_week)
                    weekly_monthly_returns[agent]["weekly"] = self.compute_weekly_monthly_returns(weekly_returns)

        if completed_months:
            last_month = completed_months[-1]  # Last complete month
            if self.date == last_month[-1]:    # Only calculate on the last day of the month
                for agent in ['quants', 'signals', 'decision']:
                    monthly_returns = get_agent_returns(agent, last_month)
                    weekly_monthly_returns[agent]["monthly"] = self.compute_weekly_monthly_returns(monthly_returns)

        return weekly_monthly_returns


    def run(self):
        self.load_eval_data()

        # Step 1: compute everything once
        accuracy = self.evaluate_accuracy()
        updated_portfolio_values = self.compute_portfolio_value()
        daily_returns = self.compute_daily_portfolio_returns()
        regret = self.compute_regret()
        cumulative_return = self.compute_cumulative_return_since_day_one()
  
        if self.date not in self.performance_log:
            self.performance_log[self.date] = {} 
        
        self.performance_log[self.date]["daily_returns"] = daily_returns

        sharpe_ratios = self.calculate_sharpe_ratios()
        if sharpe_ratios is None:
            print(f"Error: Sharpe Ratios not calculated for {self.date}")
            return None
             

        weekly_and_monthly_returns = self.calculate_weekly_and_monthly_returns()

        self.performance_log[self.date] = {
            "decision_allocation": self.get_allocation("decision"),
            "accuracy": accuracy,
            "daily_returns": daily_returns,
            "portfolio_values": updated_portfolio_values,
            "regret": regret,
            "cumulative_return": cumulative_return,
            "sharpe_ratio": sharpe_ratios,
            "returns": weekly_and_monthly_returns
        }

        #  Step 3: generate feedback
        feedback = self.build_prompt(
            btc_return=self.daily_data["btc_return_daily"],
            returns=daily_returns,
            regret=regret,
            accuracy=accuracy,
            quants_reasoning=self.quant_output["reasoning"],
            signals_reasoning=self.signal_output["reasoning"],
            decision_output=self.decision_output,
            current_portfolio=updated_portfolio_values
        )

        return feedback, self.performance_log


# if __name__ == "__main__":
#     # Path to the state folder (adjust this to your directory)
#     state_folder = os.path.join("state")

#     # Initialize an empty performance log (this will be used across all days)
#     # performance_log = open(os.path.join("state","2024-07-06.json"))
#     performance_log = {} 

#     # List all state files (sorted by date)
#     state_files = sorted([f for f in os.listdir(state_folder) if f.endswith(".json")])

#     for state_file in state_files:
#         state_file_path = os.path.join(state_folder, state_file)

#         # Load the state file for the current day
#         with open(state_file_path, "r") as f:
#             state_data = json.load(f)

#         # Extract necessary data from the state file
#         date = state_data["date"]
#         quants_output = state_data["quants_output"]
#         quants_private = state_data["quants_private"]
#         signals_output = state_data["signals_output"]
#         signals_private = state_data["signals_private"]
#         decision_output = state_data["decision_output"]

#         # Initialize the ReflectAgent with the current date's data
#         reflect = ReflectAgent(
#             date=date,
#             Quant_output=quants_output,
#             Quant_portfolio=quants_private,
#             Signal_output=signals_output,
#             Signal_portfolio=signals_private,
#             decision_output=decision_output,
#             performance_log=performance_log  # Continuously updating the same log
#         ) 

#         reflect.load_eval_data()

#         # Step 1: compute everything once
#         accuracy = reflect.evaluate_accuracy()
#         daily_returns = reflect.compute_daily_returns()
#         updated_portfolio_values = reflect.compute_portfolio_value()
#         regret = reflect.compute_regret()
#         cumulative_return = reflect.compute_cumulative_return_since_day_one()

#         if reflect.date not in reflect.performance_log:
#             reflect.performance_log[reflect.date] = {} 
        
#         reflect.performance_log[reflect.date]["daily_returns"] = daily_returns

#         sharpe_ratios = reflect.calculate_sharpe_ratios()
#         weekly_and_monthly_returns = reflect.calculate_weekly_and_monthly_returns()

#         reflect.performance_log[reflect.date] =  {
#             "decision_allocation": reflect.get_allocation("decision"),
#             "accuracy": accuracy,
#             "daily_returns": daily_returns,
#             "portfolio_values": updated_portfolio_values,
#             "regret": regret,
#             "cumulative_return": cumulative_return,
#             "sharpe_ratio": sharpe_ratios,
#             "Returns": weekly_and_monthly_returns
#         }

#     final_log_path = os.path.join("trial_states", "final_performance_log.json")
#     with open(final_log_path, "w") as f:
#         json.dump(performance_log, f, indent=4)
    



#     print(f"✅ Final Performance Log for All Days saved to: {final_log_path}")
    




                


        


        




