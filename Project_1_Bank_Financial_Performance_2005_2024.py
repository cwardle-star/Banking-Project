import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Bank of America": "BAC.csv",
    "Citigroup": "CITIGROUP.csv",
    "Wells Fargo": "WFC.csv",
    "JPMorgan Chase": "JPM.csv",
}

dfs = []
for bank, path in files.items():
    df = pd.read_csv(path)
    df["Bank"] = bank
    dfs.append(df)

combo = pd.concat(dfs, ignore_index=True)

for c in ["Year", "ROE"]:
    combo[c] = pd.to_numeric(combo[c], errors="coerce")

plt.figure(figsize=(10, 6))

for bank, g in combo.groupby("Bank"):

    g = g.sort_values("Year")

    plt.plot(g["Year"], g["ROE"], label=bank)

plt.xlabel("Year")
plt.ylabel("ROE")

plt.title("Return on Equity (2005–2024)")

plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 6))
for bank, g in combo.groupby("Bank"):
    g = g.sort_values("Year")
    plt.plot(g["Year"], g["ROA"], label=bank)

plt.xlabel("Year")
plt.ylabel("ROA")
plt.title("Return on Assets (2005–2024)")
plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for bank, g in combo.groupby("Bank"):
    g = g.sort_values("Year")

    if "Profit Margin" in g.columns:
        plt.plot(g["Year"], g["Profit Margin"], label=bank)

plt.xlabel("Year")
plt.ylabel("Profit Margin")
plt.title("Profit Margin (2005–2024)")
plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for bank, g in combo.groupby("Bank"):
    g = g.sort_values("Year")
    if "Asset Turnover" in g.columns:
        plt.plot(g["Year"], g["Asset Turnover"], label=bank)

plt.xlabel("Year")
plt.ylabel("Asset Turnover (Revenue / Assets)")
plt.title("Asset Turnover (2005–2024)")
plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for bank, g in combo.groupby("Bank"):
    g = g.sort_values("Year")
    if "Equity_to_Assets" in g.columns:
        plt.plot(g["Year"], g["Equity_to_Assets"], label=bank)

plt.xlabel("Year")
plt.ylabel("Equity-to-Assets Ratio")
plt.title("Equity-to-Assets Ratio (2005–2024)")
plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for bank, g in combo.groupby("Bank"):
    g = g.sort_values("Year")

    plt.plot(g["Year"], g["Total Revenue"], label=bank)

plt.xlabel("Year")
plt.ylabel("Revenue (USD Millions)")
plt.title("Total Revenue (2005–2024)")
plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for bank, g in combo.groupby("Bank"):
    g = g.sort_values("Year")
    plt.plot(g["Year"], g["Net Income"], label=bank)

plt.xlabel("Year")
plt.ylabel("Net Income (USD Millions)")
plt.title("Net Income (2005–2024)")
plt.legend()

plt.axvspan(2008, 2009, alpha=0.1)
plt.axvspan(2020, 2020.9, alpha=0.1)
plt.axvspan(2022, 2024, alpha=0.05)

plt.tight_layout()
plt.show()
