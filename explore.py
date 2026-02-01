from datasets import load_dataset

# Load a specific table (e.g., feedbacks)
feedbacks = load_dataset("Nutanix/codereview-dataset", "default")

# Access data (uses 'test' split)
print(len(feedbacks['test']))  # e.g., 460 rows
print(feedbacks['test'].features)  # Shows column names and types

# View first example
print(feedbacks['test'][0])  # Dict with all columns for row 0