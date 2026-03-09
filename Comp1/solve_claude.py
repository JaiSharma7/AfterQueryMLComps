#!/usr/bin/env python3
"""
SEC 10-K Risk Category Classifier using Claude API
Usage: ANTHROPIC_API_KEY=sk-ant-... python solve_claude.py
"""
import pandas as pd
import numpy as np
import anthropic
import time
import os
import json

LABELS = [
    "competitive_risk", "cyber_technology_risk", "financial_credit_risk",
    "litigation_risk", "market_risk", "operational_risk",
    "other_risk", "regulatory_risk", "strategic_risk"
]

LABEL_DESC = {
    "competitive_risk": "Competition, market share, pricing, new entrants, consumer preferences",
    "cyber_technology_risk": "Cybersecurity, data breaches, IT failures, tech infrastructure",
    "financial_credit_risk": "Credit risk, debt, liquidity, cash flow, impairment, credit ratings",
    "litigation_risk": "Lawsuits, legal proceedings, IP disputes, class actions, settlements",
    "market_risk": "Stock/commodity price volatility, interest rates, FX, macroeconomic conditions",
    "operational_risk": "Supply chain, manufacturing, workforce, natural disasters, third-party vendors",
    "other_risk": "Doesn not clearly fit other categories",
    "regulatory_risk": "Government regulations, compliance, legislation, tax policy, trade restrictions",
    "strategic_risk": "M&A, international expansion, restructuring, geopolitical risk"
}

def make_system_prompt():
    label_lines = "
".join(f"- {k}: {v}" for k, v in LABEL_DESC.items())
    return f"""You are a financial analyst specializing in SEC 10-K filings. 
Classify each sentence from a company risk factor disclosure into exactly one of these categories:
{label_lines}

Respond ONLY with a JSON array of labels (strings), one per sentence, in the same order as input.
Use only the exact label names listed above."""

def classify_batch(client, sentences, ticker, batch_num=0):
    numbered = "
".join(f"{i+1}. [{ticker}] {s}" for i, s in enumerate(sentences))
    prompt = f"Classify these {len(sentences)} sentences:

{numbered}

Respond with a JSON array of {len(sentences)} labels."
    
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=make_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            text = msg.content[0].text.strip()
            # Parse JSON
            if text.startswith("["):
                labels = json.loads(text)
            else:
                # Extract JSON array from text
                start = text.index("[")
                end = text.rindex("]") + 1
                labels = json.loads(text[start:end])
            
            # Validate labels
            valid_labels = []
            for lbl in labels:
                lbl = lbl.strip().lower().replace(" ", "_")
                if lbl in LABELS:
                    valid_labels.append(lbl)
                else:
                    # Find closest match
                    valid_labels.append("other_risk")
            return valid_labels
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return ["other_risk"] * len(sentences)

def main():
    train = pd.read_csv(r"C:/Users/jaish/Downloads/train.csv")
    test = pd.read_csv(r"C:/Users/jaish/Downloads/test.csv")
    
    print(f"Test set: {len(test)} rows")
    
    client = anthropic.Anthropic()
    
    all_predictions = []
    batch_size = 30
    
    for i in range(0, len(test), batch_size):
        batch = test.iloc[i:i+batch_size]
        sentences = batch["sentence"].fillna("").tolist()
        ticker = batch["ticker"].iloc[0]
        
        print(f"Batch {i//batch_size + 1}/{(len(test)-1)//batch_size + 1} (rows {i}-{min(i+batch_size, len(test))-1}, ticker={ticker})")
        preds = classify_batch(client, sentences, ticker, i//batch_size)
        all_predictions.extend(preds)
        
        # Small delay to avoid rate limiting
        if i + batch_size < len(test):
            time.sleep(0.5)
    
    submission = pd.DataFrame({"id": test["id"], "risk_category": all_predictions})
    out_path = r"C:/Users/jaish/Downloads/submission_claude.csv"
    submission.to_csv(out_path, index=False)
    print(f"
Saved {len(submission)} predictions to {out_path}")
    print("Distribution:")
    print(submission["risk_category"].value_counts())

if __name__ == "__main__":
    main()
