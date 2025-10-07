#!/usr/bin/env python3
"""
Quick visual comparison of encoder performance.
Run after encoders are trained (encoder_comparison_mode=True).
"""
import json, pandas as pd, matplotlib.pyplot as plt

RES_FILE = 'experiments/encoder_results/comparison.json'

def main():
    with open(RES_FILE) as f:
        data = json.load(f)
    df = pd.DataFrame(data).T
    df = df.sort_values('roc_auc', ascending=False)

    print(df.round(4))

    df[['roc_auc', 'pr_auc', 'f1_score']].plot.bar(figsize=(10,6))
    plt.ylabel('Score')
    plt.title('Encoder Performance Comparison')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
