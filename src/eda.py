from pathlib import Path

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import seaborn as sns

from inspect_data import load_data

OUT = Path("screenshots/eda")

OUT.mkdir(parents=True, exist_ok=True)


df = load_data()


# Target balance

sns.countplot(data=df, x="target")

plt.title("Heart Disease Target Distribution")

plt.tight_layout()

plt.savefig(OUT / "target_balance.png")

plt.close()


# Age distribution

sns.histplot(data=df, x="age", hue="target", bins=20, kde=True)

plt.title("Age Distribution by Target")

plt.tight_layout()

plt.savefig(OUT / "age_distribution.png")

plt.close()


# Cholesterol by target

sns.boxplot(data=df, x="target", y="chol")

plt.title("Cholesterol by Target")

plt.tight_layout()

plt.savefig(OUT / "cholesterol_by_target.png")

plt.close()


# Correlation heatmap

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(OUT / "correlation_heatmap.png")

plt.close()


print("Dataset shape:", df.shape)

print("\nMissing values:\n", df.isna().sum())

print("\nTarget distribution:\n", df["target"].value_counts())

print(f"\nEDA plots saved in {OUT}")
