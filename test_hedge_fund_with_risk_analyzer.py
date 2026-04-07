# test_hedge_fund_with_risk_analyzer.py

# This script is designed to analyze risks associated with hedge funds.

class HedgeFund:
    def __init__(self, name, assets, risk_level):
        self.name = name
        self.assets = assets
        self.risk_level = risk_level

    def analyze_risk(self):
        if self.risk_level > 5:
            return f"{self.name} has a high risk profile."
        return f"{self.name} is within acceptable risk levels."

# Example Usage
if __name__ == '__main__':
    fund = HedgeFund('Fund A', 1000000, 7)
    print(fund.analyze_risk())