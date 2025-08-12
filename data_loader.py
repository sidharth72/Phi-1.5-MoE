from datasets import load_dataset

dataset = load_dataset("sql/create_context")

print(dataset)
print(dataset["train"].column_names)
print(dataset["train"][0])
