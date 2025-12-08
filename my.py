import pandas as pd
from apyori import apriori

# ------------- Config -------------
CSV_FILE = "Sales_April_2019.csv"
MIN_SUPPORT = 0.003
MIN_CONFIDENCE = 0.2
MIN_LIFT = 1.0
# ----------------------------------

# Load
df = pd.read_csv(CSV_FILE)

# Attempt to find the invoice/order column and the item/product column automatically.
cols = [c for c in df.columns]

def find_column(keywords):
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    return None

invoice_col = find_column(['invoice', 'order', 'basket', 'transaction', 'tid'])
item_col = find_column(['item', 'product', 'description', 'name', 'stock'])

# If we couldn't automatically detect, show columns and raise helpful error.
if invoice_col is None or item_col is None:
    raise RuntimeError(
        "Couldn't autodetect invoice/order and/or item/product columns.\n"
        f"Columns found: {cols}\n"
        "Please check column names and either rename them or set invoice_col/item_col manually."
    )

# Drop columns if they exist (safe)
cols_to_drop = ["InvoiceNo", "CustomerID", "InvoiceDate", "Price"]
existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
if existing_drop_cols:
    df = df.drop(columns=existing_drop_cols)

# Keep only rows where invoice and item are not null
df = df.dropna(subset=[invoice_col, item_col])

# Normalize item values to strings and strip whitespace
df[item_col] = df[item_col].astype(str).str.strip()

# Group into transactions: list of items per invoice/order
transactions = df.groupby(invoice_col)[item_col].apply(lambda x: [i for i in x.tolist() if i != 'nan' and i != 'None']).tolist()

# Sanity check: if no transactions, raise helpful message
if len(transactions) == 0:
    raise RuntimeError("No transactions found after grouping. Check the invoice/item columns and any NaN cleaning.")

# Run Apriori
rules = apriori(
    transactions,
    min_support=MIN_SUPPORT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT
)

results = list(rules)

# Print nicely
print(f"\nDetected invoice column: {invoice_col}")
print(f"Detected item column: {item_col}")
print(f"Number of transactions: {len(transactions)}")
print("\n===== ASSOCIATION RULES =====\n")

if not results:
    print("No rules found with the current thresholds.")
    print(f"Try lowering min_support (currently {MIN_SUPPORT}), min_confidence (currently {MIN_CONFIDENCE}), or min_lift.")
else:
    for r in results:
        print(f"Support: {r.support:.6f}   (count ≈ {r.support * len(transactions):.1f})")
        for stat in r.ordered_statistics:
            base = list(stat.items_base)
            add = list(stat.items_add)
            # only print meaningful rules (non-empty antecedent -> consequent)
            if base and add:
                print(f"Rule: {base}  →  {add}")
                print(f"  Confidence: {stat.confidence:.6f}")
                print(f"  Lift:       {stat.lift:.6f}")
                print("-" * 50)
