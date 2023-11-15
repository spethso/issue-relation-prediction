import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

# Read the dataset
training_data_set = pd.read_csv('corpus/X_train.csv')

# Read the relations
relations = np.load('corpus/y_train.npy', allow_pickle=True)

# define a list of issue relation pairs
issue_pair_relation_tupels = []

# Function to map the relation to a number. The relation is given as list of form [0, 0, 0, 0, 1] where the 1 indicates the relation.
# In the given example, the relation is 4.
def map_relation_to_number(relation):
    for i in range(len(relation)):
        if relation[i] == 1:
            return i


def create_issue_pair_relation_tupels(training_data_set, relations):
    # the CSV contains three columns: one for the index, one for IssueA and one for IssueB.
    # Create a tupel for the issue pair and their relations. The IssueA and IssueB should be combined with a <s>/<s> token in between. 
    # The relation should be converted to a number.
    for i in range (len(training_data_set)):
        issueA = training_data_set['IssueA'][i]
        issueB = training_data_set['IssueB'][i]

        # Combine the issues with a <s>/<s> token in between
        combined = f'{issueA} </s></s>{issueB}'
        # Convert the relation to a number
        relation = map_relation_to_number(relations[i])
        # Create the tupel
        tupel = (combined, relation)
        # store the tupel in a list of issue relation pairs
        issue_pair_relation_tupels.append(tupel)

create_issue_pair_relation_tupels(training_data_set, relations)

# Print the issue relation pairs
# print(issue_pair_relation_tupels[:1]) 



# Define the dataset class
class IssueRelationDataset(Dataset):
    def __init__(self, issue_pair_relation_tupels, tokenizer):
        self.issue_pair_relation_tupels = issue_pair_relation_tupels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.issue_pair_relation_tupels)

    def __getitem__(self, idx):
        issue_pair_relation_tupel = self.issue_pair_relation_tupels[idx]
        issue_relation_pair_encoded = self.tokenizer.encode_plus(
            issue_pair_relation_tupel[0],
            add_special_tokens=True,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(issue_relation_pair_encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(issue_relation_pair_encoded['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(issue_relation_pair_encoded['token_type_ids'], dtype=torch.long),
            'label': torch.tensor(issue_pair_relation_tupel[1], dtype=torch.long)
        }

# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the tokenizer and the model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)

# Define the dataset and the dataloader
dataset = IssueRelationDataset(issue_pair_relation_tupels, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define the optimizer and the loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# Train the model
for epoch in range(3):
    for batch in dataloader:
        print('Healthcheck')
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the model and store in model.pt
model.save_pretrained('model')
torch.save(model, 'model/model.pt')


