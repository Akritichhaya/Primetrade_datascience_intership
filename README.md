# Trader Performance vs Market Sentiment Analysis
## Primetrade.ai — Data Science Intern Assignment

**Submitted by:** Akriti Chhaya
**Email:** theakritichhaya@gmail.com

---

## Setup & How to Run

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
jupyter notebook trader_analysis.ipynb
```

---

## Methodology

### Part A — Data Preparation
- Loaded both datasets — documented shape, missing values, duplicates
- Converted timestamps and aligned at daily level
- Created key metrics: daily PnL, win rate, leverage, trade size, long/short ratio

### Part B — Analysis
- Greed days show 2.3x higher avg PnL vs Fear days (p < 0.05)
- Traders use 89% more leverage on Greed days
- Long bias shifts from 40% Fear to 61% Greed days
- Segmented traders by leverage, frequency, performance

### Part C — Strategies
1. Cap leverage at 3x on Fear days
2. Infrequent traders avoid Fear days

---

## Key Insights
1. Greed days produce 2.3x higher PnL (statistically significant)
2. Traders use 89% more leverage on Greed days
3. Long bias increases from 40% to 61% on Greed days

*Akriti Chhaya | theakritichhaya@gmail.com*
