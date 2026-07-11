from datasets import load_dataset

print("Loading Wikipedia dataset...")

wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train[:10000]"
)

# Print the column names
print(wiki.column_names)

# Print the first article
print(wiki[0])