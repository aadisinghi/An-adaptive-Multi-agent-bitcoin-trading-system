import pandas as pd
import os
import json

class LongTermFeedbackEvaluator:

    def __init__(self, performance_log, date):
        self.performance_log = performance_log
        self.date = date
        self.long_term_feedback = {}
        self.suggestions = {
            "regret": {
                "quants": {
                    "positive": {
                        "mild": "Your strategy outperformed the baseline but remains inconsistent. Focus on improving indicator accuracy and avoiding false signals.",
                        "strong": "Your strategy outperformed the baseline frequently, effectively capturing profitable trends. Maintain your reasoning but ensure they adapt to changing conditions.",
                        "exceptional": "Your strategy dominated the baseline, achieving superior performance almost every day. Continue monitoring indicator efficiency, but avoid unnecessary changes."
                    },
                    "negative": {
                        "mild": "Your strategy underperformed the baseline on multiple days. This indicates that some indicators may be unreliable or overly reactive. Review indicator sensitivity and remove noise-heavy metrics.",
                        "strong": "Frequent underperformance against the baseline. Your indicators may be failing to capture profitable trends. Prioritize stable, trend-following metrics over volatile ones.",
                        "exceptional": "Severe underperformance. Your strategy rarely outperforms the baseline. Reconstruct your indicator selection, focusing on those with a proven track record of profitability."
                    }
                },
                "signals": {
                    "positive": {
                        "mild": "Your sentiment signals provided slightly better returns than the baseline but lacked consistency. Refine your source list and prioritize high-confidence data.",
                        "strong": "Your sentiment signals consistently outperformed the baseline, accurately capturing market trends. Maintain your source list, but test for even higher-quality data.",
                        "exceptional": "Your sentiment signals delivered superior results, consistently beating the baseline. Continue monitoring source quality to ensure sustained performance."
                    },
                    "negative": {
                        "mild": "Your sentiment signals underperformed on multiple days. This suggests that some sources are unreliable. Filter out low-quality sources and focus on trusted data.",
                        "strong": "Your sentiment signals frequently failed to outperform the baseline, indicating misleading or conflicting data. Re-evaluate your sources and ensure your signals reflect clear market sentiment.",
                        "exceptional": "Severe underperformance. Your sentiment signals rarely beat the baseline, suggesting they are fundamentally flawed. Reconstruct your source list and strengthen your filtering logic."
                    }
                },
                "decision": {
                    "positive": {
                        "mild": "Your decisions slightly outperformed the baseline but lacked strong returns. Improve allocation logic by weighting higher-confidence signals.",
                        "strong": "Your decisions consistently outperformed the baseline, effectively balancing risk and reward. Maintain your approach, but explore slight adjustments for higher returns.",
                        "exceptional": "Your decisions dominated the baseline, delivering superior returns with balanced risk. Maintain your current strategy but remain vigilant for changing conditions."
                    },
                    "negative": {
                        "mild": "Your decisions underperformed the baseline on several days. This suggests weak decision logic. Prioritize high-confidence inputs and avoid overreacting to short-term signals.",
                        "strong": "Your decisions frequently failed to beat the baseline, indicating poor allocation logic. Re-evaluate how you weigh quants and signals and reduce exposure to low-confidence inputs.",
                        "exceptional": "Severe underperformance. Your decision logic is fundamentally flawed. Reconstruct your allocation strategy to focus on reliable signals and reduce risky allocations."
                    }
                }
            },

            "sharpe_ratio": {
                "quants": {
                    "positive": {
                        "mild": "Your strategy has slightly better risk-adjusted performance, but some days are highly volatile. Review your allocation strategy.",
                        "strong": "Good risk management! Your quant strategy maintains high returns with low volatility.",
                        "exceptional": "Outstanding risk-adjusted performance! Your allocations are consistently profitable with minimal risk."
                    },
                    "negative": {
                        "mild": "Your strategy has slightly higher volatility than expected. Identify which indicators contribute to excessive risk.",
                        "strong": "High risk exposure in strategy. Your allocations fluctuate significantly. Consider reducing position size.",
                        "exceptional": "Severe volatility. Your allocations are erratic and lead to high losses. Redesign your strategy to prioritize stability."
                    }
                },
                "signals": {
                    "positive": {
                        "mild": "Your sentiment analysis generally balances risk and reward. Maintain source reliability and avoid reacting to low-confidence signals.",
                        "strong": "Good strategy! Your signals consistently capture profitable sentiment shifts with controlled risk.",
                        "exceptional": "Exceptional risk-adjusted sentiment analysis! Your signals are highly accurate with minimal noise."
                    },
                    "negative": {
                        "mild": "Your sentiment analysis has slightly higher volatility. Reassess which sources are most reliable.",
                        "strong": "High risk exposure in sentiment analysis. You are reacting to conflicting or volatile sources. Reduce exposure to unreliable data.",
                        "exceptional": "Severe volatility in sentiment. Your signals are misleading and cause erratic performance. Re-evaluate your filtering logic."
                    }
                },
                "decision": {
                    "positive": {
                        "mild": "Your decision allocations are generally balanced but can be slightly optimized. Prioritize high-confidence inputs.",
                        "strong": "Good decision-making! Your portfolio maintains high returns with controlled risk.",
                        "exceptional": "Exceptional decision-making! Your allocations achieve high returns with minimal volatility."
                    },
                    "negative": {
                        "mild": "Your decision-making has slightly higher risk exposure. Re-assess how you weigh quants and signals.",
                        "strong": "High risk in decisions. You are overexposing the portfolio to volatile allocations. Reconsider your risk thresholds.",
                        "exceptional": "Severe risk in decisions. Your allocations are erratic and consistently lead to losses. Redesign your decision logic."
                    }
                }
            },

            "returns": {
                "quants": {
                    "positive": {
                        "mild": "Your quant-based allocations are slightly profitable. However, some market states were misclassified, leading to minor losses. Refine your classification approach.",
                        "strong": "Excellent quant-based returns! Your market classifications are accurate, and your allocations capture profitable trends. Maintain your strategy.",
                        "exceptional": "Outstanding quant-based performance! Your allocations consistently generate high profits, demonstrating strong market understanding."
                    },
                    "negative": {
                        "mild": "Your returns are slightly below the baseline. This may be due to misclassifying sideways markets as trending. Re-evaluate your market classification logic.",
                        "strong": "Your returns are significantly below the baseline. You may be overreacting to minor fluctuations or following misleading trends. Review your indicator weighting.",
                        "exceptional": "Severe underperformance. Your market classifications are frequently wrong, leading to heavy losses. Consider redesigning your strategy."
                    }
                },
                "signals": {
                    "positive": {
                        "mild": "Your sentiment-based returns are slightly profitable. However, some low-confidence sources may be affecting performance. Review your sentiment weighting.",
                        "strong": "Excellent sentiment-based returns! Your signals accurately capture market mood, leading to consistent profits. Maintain your approach.",
                        "exceptional": "Outstanding sentiment-based performance! Your signals perfectly capture profitable sentiment shifts, with minimal noise."
                    },
                    "negative": {
                        "mild": "Your sentiment-based returns are slightly below the baseline. Consider filtering out low-quality sources that produce misleading signals.",
                        "strong": "Your sentiment-based returns are significantly below the baseline. This may be due to conflicting signals or unreliable sources. Refine your source list.",
                        "exceptional": "Severe underperformance in sentiment analysis. Your signals consistently lead to losses. Re-evaluate your sentiment analysis logic and filter criteria."
                    }
                },
                "decision": {
                    "positive": {
                        "mild": "Your returns are slightly profitable, but some allocations missed potential gains. Refine your weighting of signals and quants.",
                        "strong": "Excellent returns! Your allocations consistently capture profitable trends, balancing quants and signals.",
                        "exceptional": "Outstanding performance! Your portfolio consistently achieves high profits with balanced exposure."
                    },
                    "negative": {
                        "mild": "Your returns are slightly below the baseline. This may be due to overreacting to low-confidence inputs.",
                        "strong": "Your returns are significantly below the baseline. Your allocations often misjudge market direction. Re-assess how you balance quants and signals.",
                        "exceptional": "Severe underperformance in decision-making. Your allocations consistently lose value. Re-evaluate your decision logic entirely."
                    }
            }}} 
    
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


    def evaluate_feedback(self, period_dates, period="weekly"):

        self.load_eval_data()
        feedback = {"quants": "", "signals": "", "decision": ""}

        # Get baseline values strictly from allowed keys
        baseline_sharpe = float(self.baseline_data['sharpe_weekly'])
        baseline_return = float(self.baseline_data['return_7d_avg'])

        for agent in ['quants', 'signals', 'decision']:
            # Calculate metrics for the period
            daily_regret = [
                self.performance_log[date]["regret"][agent] 
                for date in period_dates 
                if "regret" in self.performance_log[date]
            ]
            
            sharpe_ratio = float(self.performance_log[period_dates[-1]]["sharpe_ratio"][agent].get(period, None))
            returns = float(self.performance_log[period_dates[-1]]["returns"][agent].get(period, None))

            # Calculate Regret Performance
            total_days = len(daily_regret)
            outperforming_days = sum(1 for r in daily_regret if r > 0)
            underperforming_days = sum(1 for r in daily_regret if r < 0)
            regret_ratio = ((outperforming_days-underperforming_days)/total_days) * 100 

            feedback[agent] = self.generate_agent_feedback(
                agent,
                regret_ratio,
                sharpe_ratio,
                baseline_sharpe,
                returns,
                baseline_return
            )

        return feedback
    

    def generate_agent_feedback(self,agent, regret_ratio, sharpe_ratio, baseline_sharpe, returns, baseline_return):
        feedback = " Regret shows how often you defeated the returns by baseline, returns shows how much better or worse your results were, and Sharpe ratio shows how much better or worse your risk-adjusted performance was, each calculated as a percentage difference from the baseline."
        # Regret Feedback (Outperforming and Underperforming)
        if regret_ratio is not None: 
            if regret_ratio > 0:
                if regret_ratio >= 70:
                    feedback += f"Regret (Outperformance): {regret_ratio}. {self.suggestions['regret'][agent]['positive']['exceptional']} "
                elif regret_ratio >= 40:
                    feedback += f"Regret (Outperformance): {regret_ratio}. {self.suggestions['regret'][agent]['positive']['strong']} "
                else:
                    feedback += f"Regret (Outperformance): {regret_ratio}. {self.suggestions['regret'][agent]['positive']['mild']} "

            if regret_ratio <= 0:
                if regret_ratio <= -70:
                    feedback += f"Regret (Underperformance): {regret_ratio}. {self.suggestions['regret'][agent]['negative']['exceptional']} "
                elif regret_ratio < -40:
                    feedback += f"Regret (Underperformance): {regret_ratio}. {self.suggestions['regret'][agent]['negative']['strong']} "
                else:
                    feedback += f"Regret (Underperformance): {regret_ratio}. {self.suggestions['regret'][agent]['negative']['mild']} "

        # Sharpe Ratio Feedback
        if sharpe_ratio is not None:
            sharpe_diff = round((sharpe_ratio - baseline_sharpe) / baseline_sharpe * 100, 2) if baseline_sharpe != 0 else 0
            
            if sharpe_diff>0: 
                if sharpe_diff >= 70:
                    feedback += f"Sharpe Ratio: +{sharpe_diff}%. {self.suggestions['sharpe_ratio'][agent]['positive']['exceptional']} "
                elif sharpe_diff >= 40:
                    feedback += f"Sharpe Ratio: +{sharpe_diff}%. {self.suggestions['sharpe_ratio'][agent]['positive']['strong']} "
                else:
                    feedback += f"Sharpe Ratio: +{sharpe_diff}%. {self.suggestions['sharpe_ratio'][agent]['positive']['mild']} "
            
            if sharpe_diff<=0: 
                if sharpe_diff <= -70:
                    feedback += f"Sharpe Ratio: {sharpe_diff}%. {self.suggestions['sharpe_ratio'][agent]['negative']['exceptional']} "
                elif sharpe_diff <= -40:
                    feedback += f"Sharpe Ratio: {sharpe_diff}%. {self.suggestions['sharpe_ratio'][agent]['negative']['strong']} "
                else:
                    feedback += f"Sharpe Ratio: {sharpe_diff}%. {self.suggestions['sharpe_ratio'][agent]['negative']['mild']} "

        # Returns Feedback
        if returns is not None:
            returns_diff = round((returns - baseline_return) / baseline_return * 100, 2) if baseline_return != 0 else 0
            if returns_diff>0: 
                if returns_diff >= 70:
                    feedback += f"Returns: +{returns_diff}%. {self.suggestions['returns'][agent]['positive']['exceptional']} "
                elif returns_diff >= 40:
                    feedback += f"Returns: +{returns_diff}%. {self.suggestions['returns'][agent]['positive']['strong']} "
                else:
                    feedback += f"Returns: +{returns_diff}%. {self.suggestions['returns'][agent]['positive']['mild']} "
            
            if returns_diff<=0: 
                if returns_diff <= -70:
                    feedback += f"Returns: {returns_diff}%. {self.suggestions['returns'][agent]['negative']['exceptional']} "
                elif returns_diff <= -40:
                    feedback += f"Returns: {returns_diff}%. {self.suggestions['returns'][agent]['negative']['strong']} "
                else:
                    feedback += f"Returns: {returns_diff}%. {self.suggestions['returns'][agent]['negative']['mild']} "

        return feedback
 

    def generate_long_term_feedback(self):
        last_stored_feedback_weekly = {} 

        sorted_dates = sorted(self.performance_log.keys())

        # Calculate non-overlapping weeks (7-day blocks) and months (30-day blocks)
        completed_weeks = [sorted_dates[i:i+7] for i in range(0, len(sorted_dates), 7) if len(sorted_dates[i:i+7]) == 7]

        prev_date = None
        current_index = sorted_dates.index(self.date)
        if current_index > 0:
            prev_date = sorted_dates[current_index - 1]

        if prev_date:
                last_stored_feedback_weekly = self.performance_log.get(prev_date, {}).get("last_long_term_feedback_weekly", {})

        # Only trigger feedback on the last day of the week/month
        if completed_weeks and self.date == completed_weeks[-1][-1]:
            last_week = completed_weeks[-1]
            self.long_term_feedback["weekly"] = self.evaluate_feedback(last_week, period="weekly")
            self.performance_log[self.date]["last_long_term_feedback_weekly"] = self.long_term_feedback.get("weekly",{})
        
        else:
            self.long_term_feedback["weekly"]= last_stored_feedback_weekly if last_stored_feedback_weekly else {} 
            self.performance_log[self.date]["last_long_term_feedback_weekly"] = self.long_term_feedback["weekly"] 
        
        return self.long_term_feedback , self.performance_log


    def run_long_term_feedback(self):
        print(f"📝 Running Long-Term Feedback for {self.date}...")

        # Generate and store long-term feedback
        feedback, self.performance_log = self.generate_long_term_feedback()
        # print("long term feedback",feedback)
        # print("performance log", self.performance_log)
        
        print(f"✅ Long-Term Feedback generated for {self.date}.")
        return feedback, self.performance_log


