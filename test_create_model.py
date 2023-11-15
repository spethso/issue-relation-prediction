import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch


text_relation_tupels = (
    []
)  # Should contain tuples of form (text, relation), where the relation is a number from 0 to 4

# Create 1000 entries where 'lorem ipsum' is relation 0, 'dolor sit' is relation 1, 'amet' is relation 2, 'consectetur' is relation 3 and 'adipiscing elit' is relation 4.
for i in range(200):
    text_relation_tupels.append(("lorem ipsum", 0))
    text_relation_tupels.append(("dolor sit", 1))
    text_relation_tupels.append(("amet", 2))
    text_relation_tupels.append(("consectetur", 3))
    text_relation_tupels.append(("adipiscing elit", 4))

# Shuffle the list
np.random.shuffle(text_relation_tupels)


# Define the dataset class
class RelationDataset(Dataset):
    def __init__(self, text_relation_tupels, tokenizer):
        self.text_relation_tupels = text_relation_tupels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_relation_tupels)

    def __getitem__(self, index):
        text, relation = self.text_relation_tupels[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )
        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(relation, dtype=torch.long),
        }


# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
# If not...
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# Load the tokenizer and the model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=5
).to(device)

dataset = RelationDataset(text_relation_tupels, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        print("Epoch: %d, Loss: %.3f" % (epoch, loss.item()))

model.save_pretrained("model")
torch.save(model, "model/model.pt")
