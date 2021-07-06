# import os
# import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split

from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


BEST_MODEL_PATH = "boolq-bert-val-loss.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

SEED = 42
def set_seed():
	np.random.seed(SEED)
	torch.random.manual_seed(SEED)
	torch.cuda.random.manual_seed(SEED)
	torch.cuda.random.manual_seed_all(SEED)

	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True


# -------------------BEGIN READ DATASET-------------------------
train_file = "boolq/train.jsonl"
dev_file = "boolq/dev.jsonl"

df_train = pd.read_json(train_file, lines=True, orient="records")
df_dev = pd.read_json(dev_file, lines=True, orient="records")

def preproc(df):
	df.answer = df.answer.astype(int)

preproc(df_train)
preproc(df_dev)
# --------------------END READ DATASET---------------------------


def encode_df(tokenizer, df, max_length, use_segment_ids):
	"""encode question, passage"""
	input_ids = []
	attention_mask = []
	segment_ids = []

	questions = df.question.values
	passages = df.passage.values
	answers = df.answer.values

	for question, passage in zip(questions, passages):
		encoded = tokenizer.encode_plus(question,
										 passage,
										 max_length=max_length,
										 pad_to_max_length=True,
										 truncation_strategy="longest_first")
		input_ids.append(encoded["input_ids"])
		attention_mask.append(encoded["attention_mask"])
		if use_segment_ids:
			segment_ids.append(encoded["token_type_ids"])

	values = [input_ids, attention_mask, answers]
	if use_segment_ids:
		values.append(segment_ids)

	tensors = (torch.tensor(v) for v in values)
	return tensors


def get_dataloaders(train_tensors, test_tensors, batch_size):
	train_dataset = TensorDataset(*train_tensors)
	test_dataset = TensorDataset(*test_tensors)

	train_val_size = len(train_dataset)
	train_size = int(0.9 * train_val_size)
	val_size = train_val_size - train_size

	torch.manual_seed(SEED)
	train_data, val_data = random_split(train_dataset, [train_size, val_size])

	train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
	val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)
	test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

	return train_loader, val_loader, test_loader


def train_loop(model, optimizer, train_iter, val_iter, num_epochs, clip):
	train_losses, val_losses = [], []
	n_train = len(train_iter)

	best_val_loss = float("inf")

	for epoch in range(1, num_epochs + 1):
		train_loss = 0.0
		model.train() #train mode

		for i, batch in enumerate(train_iter, 1):
			input_ids = batch[0].to(DEVICE)
			attention_mask = batch[1].to(DEVICE)
			labels = batch[2].to(DEVICE)
			segment_ids = batch[3].to(DEVICE) if len(batch) > 3 else None

			optimizer.zero_grad()

			outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, labels=labels)

			loss = outputs.loss

			train_loss += loss.item()

			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # сlipping gradients
			optimizer.step()

		train_loss /= n_train

		print(f"Epoch: {epoch:02}")
		print(f"\tTrain Loss: {train_loss:.3f}")

		val_loss, val_acc, val_f1 = test(model, val_iter)
		print(f"\t Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc:.3f} |  Val. F1: {val_f1:.3f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(model.state_dict(), BEST_MODEL_PATH)

		train_losses.append(train_loss)
		val_losses.append(val_loss)

	return train_losses, val_losses


def test(model, iterator, to_print=False):
	true_list = []
	pred_list = []
	n_test = len(iterator)

	test_loss = 0.0

	model.eval()
	with torch.no_grad():
		for batch in iterator:
			input_ids = batch[0].to(DEVICE)
			attention_mask = batch[1].to(DEVICE)
			labels = batch[2].to(DEVICE)
			segment_ids = batch[3].to(DEVICE) if len(batch) > 3 else None

			outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, labels=labels)
			loss = outputs.loss
			logits = outputs.logits

			test_loss += loss.item()

			preds = logits.argmax(-1)
			true_list.append(labels.detach().cpu())
			pred_list.append(preds.detach().cpu())

	test_loss /= n_test

	true_labels = torch.cat(true_list, dim=0)
	pred_labels = torch.cat(pred_list, dim=0)

	test_acc = accuracy_score(true_labels, pred_labels)
	test_f1 = f1_score(true_labels, pred_labels)

	if to_print:
		print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | Test F1: {test_f1:.3f}")

	return test_loss, test_acc, test_f1


def run_model(model_name, batch_size=32, max_length=256, n_epochs=3, lr=1e-5, clip=1, use_segment_ids=True, load_state=False):
	# load pretrained model/tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)

	if load_state:
		print("loading state...")
		model.load_state_dict(torch.load(BEST_MODEL_PATH))

	set_seed()

	# encode data
	train_tensors = encode_df(tokenizer, df_train, max_length, use_segment_ids)
	test_tensors = encode_df(tokenizer, df_dev, max_length, use_segment_ids)

	# build loaders
	train_loader, val_loader, test_loader = get_dataloaders(train_tensors, test_tensors, batch_size)

	optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)

	print("training...")
	history = train_loop(model, optimizer, train_loader, val_loader, n_epochs, clip)

	print("---------- LAST MODEL -----------")
	test(model, test_loader, to_print=True)

	print("---------- BEST MODEL -----------")
	model.load_state_dict(torch.load(BEST_MODEL_PATH))
	test(model, test_loader, to_print=True)

	return history


if __name__ == "__main__":
	history = run_model("bert-base-uncased")
	# plot_training(*history)