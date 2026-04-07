# HF_04.26
# AI Hedge Fund con Financial Risk Analyzer Agent
# 1. Clone el repositorio
git clone https://github.com/jxsxLxrrx/HF_04.26.git
cd HF_04.26

# 2. Clone el repositorio original para copiar archivos
git clone https://github.com/virattt/ai-hedge-fund.git temp
cp -r temp/* . 
rm -rf temp

# 3. Crea la carpeta del nuevo agente
mkdir -p src/agents

# 4. Crea el archivo del nuevo agente (lo voy a dar abajo)
touch src/agents/financial_risk_analyzer.py
from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.tools.api import get_prices, prices_to_df, get_financial_metrics
from src.utils.api_key import get_api_key_from_state
import json
import numpy as np
import pandas as pd

def financial_risk_analyzer_agent(state: AgentState, agent_id: str = "financial_risk_analyzer_agent"):
    """
    Comprehensive financial risk analysis agent
    Evaluates: Liquidity, Market, Credit, Operational, and Systemic Risks
    """
    data = state["data"]
    tickers = data["tickers"]
    end_date = data["end_date"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    risk_analysis = {}
    
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing financial risk...")
        
        try:
            prices = get_prices(
                ticker=ticker,
                start_date=data["start_date"],
                end_date=end_date,
                api_key=api_key,
            )
            
            if not prices:
                risk_analysis[ticker] = create_default_risk_analysis(ticker)
                continue
            
            prices_df = prices_to_df(prices)
            financial_metrics = get_financial_metrics(
                ticker=ticker, end_date=end_date, period="ttm", limit=8, api_key=api_key
            )
            
            # Calculate risk dimensions
            liquidity_risk = calculate_liquidity_risk(prices_df)
            market_risk = calculate_market_risk(prices_df)
            credit_risk = calculate_credit_risk(financial_metrics) if financial_metrics else {}
            operational_risk = calculate_operational_risk(financial_metrics) if financial_metrics else {}
            systemic_risk = calculate_systemic_risk(prices_df)
            
            # Overall score
            overall_score, risk_level = calculate_overall_risk_score(
                liquidity_risk, market_risk, credit_risk, operational_risk, systemic_risk
            )
            
            risk_analysis[ticker] = {
                "overall_risk_score": overall_score,
                "risk_level": risk_level,
                "recommendation": get_risk_recommendation(overall_score),
                "liquidity_risk": liquidity_risk,
                "market_risk": market_risk,
                "credit_risk": credit_risk,
                "operational_risk": operational_risk,
                "systemic_risk": systemic_risk,
            }
            
            progress.update_status(agent_id, ticker, f"Risk: {overall_score}/100 ({risk_level})")
            
        except Exception as e:
            risk_analysis[ticker] = create_default_risk_analysis(ticker)
    
    progress.update_status(agent_id, None, "Done")
    message = HumanMessage(content=json.dumps(risk_analysis), name=agent_id)
    
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Financial Risk Analyzer")
    
    state["data"]["analyst_signals"][agent_id] = risk_analysis
    
    return {"messages": state["messages"] + [message], "data": data}


def calculate_liquidity_risk(prices_df):
    if prices_df.empty: return {"score": 50, "status": "Unknown"}
    avg_volume = prices_df["volume"].mean() if "volume" in prices_df.columns else 0
    daily_returns = prices_df["close"].pct_change().dropna()
    volatility = daily_returns.std() * 100
    
    score = 50
    if avg_volume > 10_000_000: score -= 20
    if volatility < 2: score -= 10
    score = max(0, min(100, score))
    
    return {
        "score": int(score),
        "status": "Strong" if score < 40 else "Moderate" if score < 70 else "Weak",
        "avg_volume_millions": round(avg_volume / 1_000_000, 2),
        "volatility_percent": round(volatility, 2)
    }


def calculate_market_risk(prices_df):
    if prices_df.empty: return {"score": 50, "status": "Unknown", "beta": 1.0}
    daily_returns = prices_df["close"].pct_change().dropna()
    annual_vol = daily_returns.std() * np.sqrt(252)
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    max_drawdown = ((cumulative - running_max) / running_max).min()
    
    score = 50
    if annual_vol < 0.20: score -= 15
    if max_drawdown > -0.30: score -= 10
    score = max(0, min(100, score))
    
    return {
        "score": int(score),
        "status": "Low" if score < 35 else "Moderate" if score < 65 else "High",
        "beta": round(annual_vol / 0.15, 2),
        "annual_volatility_percent": round(annual_vol * 100, 2),
        "max_drawdown_percent": round(max_drawdown * 100, 2)
    }


def calculate_credit_risk(financial_metrics):
    if not financial_metrics: return {"score": 50, "status": "Unknown"}
    latest = financial_metrics[0]
    
    score = 50
    debt_to_equity = getattr(latest, 'debt_to_equity', None)
    if debt_to_equity and debt_to_equity < 1.0: score -= 15
    
    interest_coverage = getattr(latest, 'interest_coverage', None)
    if interest_coverage and interest_coverage > 5: score -= 10
    
    score = max(0, min(100, score))
    return {
        "score": int(score),
        "status": "Strong" if score < 40 else "Moderate" if score < 70 else "Weak",
        "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity else None
    }


def calculate_operational_risk(financial_metrics):
    if not financial_metrics: return {"score": 50, "status": "Unknown"}
    latest = financial_metrics[0]
    
    score = 50
    revenue_growth = getattr(latest, 'revenue_growth', None)
    if revenue_growth and 0.10 < revenue_growth < 0.30: score -= 15
    
    score = max(0, min(100, score))
    return {
        "score": int(score),
        "status": "Low Risk" if score > 65 else "Moderate" if score > 35 else "High Risk"
    }


def calculate_systemic_risk(prices_df):
    if prices_df.empty: return {"score": 50, "status": "Unknown"}
    daily_returns = prices_df["close"].pct_change().dropna()
    skewness = daily_returns.skew()
    
    score = 50
    if skewness < -0.5: score += 15
    
    score = max(0, min(100, score))
    return {
        "score": int(score),
        "status": "Low" if score < 40 else "Moderate" if score < 70 else "High"
    }


def calculate_overall_risk_score(liquidity, market, credit, operational, systemic):
    overall = (
        market.get("score", 50) * 0.35 +
        credit.get("score", 50) * 0.25 +
        operational.get("score", 50) * 0.20 +
        systemic.get("score", 50) * 0.15 +
        liquidity.get("score", 50) * 0.05
    )
    overall = int(overall)
    
    if overall < 30: level = "VERY LOW"
    elif overall < 45: level = "LOW"
    elif overall < 60: level = "MODERATE"
    elif overall < 75: level = "HIGH"
    else: level = "VERY HIGH"
    
    return overall, level


def get_risk_recommendation(score):
    if score < 30: return "✓ SUITABLE - Very low risk"
    elif score < 45: return "✓ SUITABLE - Low risk"
    elif score < 60: return "⚠ MODERATE - Balanced risk"
    elif score < 75: return "⚠ CAUTION - High risk"
    else: return "✗ NOT SUITABLE - Very high risk"


def create_default_risk_analysis(ticker):
    return {
        "overall_risk_score": 50,
        "risk_level": "UNKNOWN",
        "recommendation": "DATA UNAVAILABLE",
        "liquidity_risk": {"score": 50, "status": "Unknown"},
        "market_risk": {"score": 50, "status": "Unknown"},
        "credit_risk": {"score": 50, "status": "Unknown"},
        "operational_risk": {"score": 50, "status": "Unknown"},
        "systemic_risk": {"score": 50, "status": "Unknown"}
    }
