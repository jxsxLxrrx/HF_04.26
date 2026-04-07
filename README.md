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
