import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score


# Read the dataset
testing_data_set = pd.read_csv("corpus/X_test.csv")

# Read the relations
relations = np.load("corpus/y_test.npy", allow_pickle=True)

# define a list of issue relation pairs
issue_pair_relation_tupels = []


# Function to map the relation to a number. The relation is given as list of form [0, 0, 0, 0, 1] where the 1 indicates the relation.
# In the given example, the relation is 4.
def map_relation_to_number(relation):
    for i in range(len(relation)):
        if relation[i] == 1:
            return i


def create_issue_pair_relation_tupels(testing_data_set, relations):
    # the CSV contains three columns: one for the index, one for IssueA and one for IssueB.
    # Create a tupel for the issue pair and their relations. The IssueA and IssueB should be combined with a <s>/<s> token in between.
    # The relation should be converted to a number.
    for i in range(len(testing_data_set)):
        issueA = testing_data_set["IssueA"][i]
        issueB = testing_data_set["IssueB"][i]

        # Combine the issues with a <s>/<s> token in between
        # combined = f'{issueA} </s></s>{issueB}'
        # Convert the relation to a number
        relation = map_relation_to_number(relations[i])
        # Create the tupel
        tupel = (issueA, issueB, relation)
        # store the tupel in a list of issue relation pairs
        issue_pair_relation_tupels.append(tupel)


create_issue_pair_relation_tupels(testing_data_set, relations)

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


# Load the pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained("model").to(device)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Define the list to store the predictions and actual relations
predictions = []
actual_relations = []

healthcheckCounter = 1

# Loop through each issue pair and predict the relation
for issueA, issueB, relation in issue_pair_relation_tupels:
    print(f"healthcheck {healthcheckCounter}")
    healthcheckCounter += 1
    # Tokenize the issue pair
    encoded_pair = tokenizer.encode_plus(
        issueA,
        issueB,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True,
    )
    # Convert the tokenized issue pair into input features
    #    input_ids = encoded_pair["input_ids"]
    input_ids = torch.tensor(
                encoded_pair["input_ids"], dtype=torch.long
    )
    # attention_mask = encoded_pair["attention_mask"]
    attention_mask = torch.tensor(
                encoded_pair["attention_mask"], dtype=torch.long
            )
    token_type_ids = torch.tensor(
                encoded_pair["token_type_ids"], dtype=torch.long
            )
    # Use the model to predict the relation
    print(input_ids)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]
        predicted_relation = torch.argmax(logits, dim=1).item()
    # Store the predicted and actual relation
    predictions.append(predicted_relation)
    actual_relations.append(relation)

# Calculate the F1 score for each relation type and the macroF1 score
f1_scores = f1_score(actual_relations, predictions, average=None)
macro_f1_score = f1_score(actual_relations, predictions, average="macro")

print(predictions[:30])
print("===========================")
print(actual_relations[:30])

# Print the results
print(f"F1 scores: {f1_scores}")
print(f"Macro F1 score: {macro_f1_score}")
