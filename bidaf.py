import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score

from torchtext.legacy.data import Field, NestedField, Dataset, TabularDataset, BucketIterator

import nltk
nltk.download("punkt")


# BEST_MODEL_PATH = "boolq-bidaf-val-loss.pt"
# BEST_MODEL_PATH = "boolq-bidaf-val-f1.pt"
BEST_MODEL_PATH = "boolq-bidaf-val-acc.pt"

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


def word_tokenize(tokens):
	return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class Loader(object):
	def __init__(self, train_batch_size, dev_batch_size, vectors, device):
		path = "boolq"
		dataset_path = os.path.join(path, "torchtext")
		train_examples_path = os.path.join(dataset_path, "train_examples.pt")
		dev_examples_path = os.path.join(dataset_path, "dev_examples.pt")

		self.CHAR = NestedField(
			Field(batch_first=True, tokenize=list, lower=True),
			tokenize=word_tokenize
		)
		self.WORD = Field(
			batch_first=True,
			lower=True,
			tokenize=word_tokenize
		)
		self.LABEL = Field(
			unk_token=None,
			sequential=False,
			use_vocab=False,
			dtype=torch.float
		)

		dict_fields = {"answer": ("answer", self.LABEL),
					   "passage": [("c_word", self.WORD),("c_char", self.CHAR)],
					   "question": [("q_word", self.WORD), ("q_char", self.CHAR)]}

		list_fields = [("answer", self.LABEL),
					   ("c_word", self.WORD), ("c_char", self.CHAR),
					   ("q_word", self.WORD), ("q_char", self.CHAR)]

		if os.path.exists(dataset_path):
			print("loading splits...")
			train_examples = torch.load(train_examples_path)
			dev_examples = torch.load(dev_examples_path)

			train_dataset = Dataset(examples=train_examples, fields=list_fields)
			dev_dataset = Dataset(examples=dev_examples, fields=list_fields)
		else:
			print("building splits...")
			train_dataset, dev_dataset = TabularDataset.splits(
				path=path,
				train="train.jsonl",
				validation="dev.jsonl",
				format="json",
				fields=dict_fields)

			os.makedirs(dataset_path)
			torch.save(train_dataset.examples, train_examples_path)
			torch.save(dev_dataset.examples, dev_examples_path)

		train_data, val_data = train_dataset.split(split_ratio=[0.9, 0.1], random_state=random.seed(SEED))

		print("building vocab...")
		self.CHAR.build_vocab(train_data)
		self.WORD.build_vocab(train_data, vectors=vectors)

		print("building iterators...")
		self.train_iter = self.create_iterator(train_data, train_batch_size, device, shuffle=True)
		self.val_iter = self.create_iterator(val_data, train_batch_size, device, shuffle=False)
		self.test_iter = self.create_iterator(dev_dataset, dev_batch_size, device, shuffle=False)

	@staticmethod
	def create_iterator(dataset, batch_size, device, shuffle):
		return BucketIterator(
			dataset,
			batch_size=batch_size,
			device=device,
			shuffle=shuffle,
			sort_key=lambda x: len(x.c_word))


class BiDAF(nn.Module):
	def __init__(
		self,
		pretrained,
		char_vocab_size,
		char_emb_dim,
		char_hidden_size,
		char_kernel_size,
		emb_dim,
		hidden_size,
		dropout,
	):
		super().__init__()

		self.char_emb_dim = char_emb_dim
		self.char_hidden_size = char_hidden_size
		self.hidden_size = hidden_size

		# 1. ------------------------------
		self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim)
		# nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

		self.char_conv = nn.Conv2d(1, char_hidden_size, (char_emb_dim, char_kernel_size))

		# 2. ------------------------------
		self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

		# 3. ------------------------------
		self.contextual_lstm = nn.LSTM(
			input_size=emb_dim + char_hidden_size,
			hidden_size=hidden_size,
			bidirectional=True,
			batch_first=True,
			dropout=dropout
		)

		# 4. ------------------------------
		self.alpha = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(hidden_size*6, 1)
		)

		# 5. ------------------------------
		self.modeling_lstm_first = nn.LSTM(
			input_size=hidden_size*8,
			hidden_size=hidden_size,
			bidirectional=True,
			batch_first=True,
			dropout=dropout
		)
		self.modeling_lstm_second = nn.LSTM(
			input_size=hidden_size*2,
			hidden_size=hidden_size,
			bidirectional=True,
			batch_first=True,
			dropout=dropout
		)

		# 6. ------------------------------
		self.lstm = nn.LSTM(
			input_size=hidden_size*10,
			hidden_size=hidden_size,
			bidirectional=True,
			batch_first=True,
			dropout=dropout
		)

		self.clf = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(hidden_size*4, 1)
		)

		self.dropout = nn.Dropout(p=dropout)

	def embed(self, batch):
		batch_size = batch.size(0)

		emb = self.char_emb(batch)
		emb = self.dropout(emb)

		emb = emb.transpose(2, 3)
		emb = emb.view(-1, self.char_emb_dim, emb.size(3)).unsqueeze(1)

		emb = self.char_conv(emb).squeeze()
		emb = F.max_pool1d(emb, emb.size(2)).squeeze()

		emb = emb.view(batch_size, -1, self.char_hidden_size)
		return emb

	def attention(self, context, question):
		tensor = torch.cat([
			context.unsqueeze(2).expand(context.size(0), context.size(1), question.size(1), -1),
			question.unsqueeze(1).expand(context.size(0), context.size(1), question.size(1), -1),
			context.unsqueeze(2) * question.unsqueeze(1)
		], dim=-1)
		s = self.alpha(tensor).squeeze()

		a = F.softmax(s, dim=2)
		context_question_attention = torch.bmm(a, question)

		b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
		question_context_attention = torch.bmm(b, context).squeeze()
		question_context_attention = question_context_attention.unsqueeze(1).expand(-1, context.size(1), -1)

		result = torch.cat([
					  context,
					  context_question_attention,
					  context * context_question_attention,
					  context * question_context_attention
		], dim=-1)
		return result

	def forward(self, batch):
		n_batch = batch.c_word.size(0)

		# 1. Character Embedding Layer
		context_char_emb = self.embed(batch.c_char)
		question_char_emb = self.embed(batch.q_char)

		# 2. Word Embedding Layer
		context_word_emb = self.word_emb(batch.c_word)
		question_word_emb = self.word_emb(batch.q_word)

		context = torch.cat([context_char_emb, context_word_emb], dim=-1)
		question = torch.cat([question_char_emb, question_word_emb], dim=-1)

		# 3. Contextual Embedding Layer
		context, _ = self.contextual_lstm(context)
		question, _ = self.contextual_lstm(question)

		# 4. Attention Flow Layer
		g = self.attention(context, question)

		# 5. Modeling Layer
		features, _ = self.modeling_lstm_first(g)
		features, _ = self.modeling_lstm_second(features)

		# 6. Output Layer
		_, features = self.lstm(torch.cat([g, features], dim=-1))
		features = torch.cat((
			features[0].permute(1, 0, 2).reshape(n_batch, self.hidden_size*2),
			features[1].permute(1, 0, 2).reshape(n_batch, self.hidden_size*2)
		), dim=1)

		logits = self.clf(features)
		return logits.flatten()


def train_loop(model, optimizer, criterion, train_iter, val_iter, num_epochs, clip):
	train_losses, val_losses = [], []
	n_train = len(train_iter)

	max_val_acc = -1

	for epoch in range(1, num_epochs + 1):
		train_loss = 0.0
		model.train() #train mode
		for batch in train_iter:
			labels = batch.answer

			optimizer.zero_grad()

			logits = model(batch)
			loss = criterion(logits, labels)
			loss.backward()
			train_loss += loss.item()

			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

			optimizer.step()

		train_loss /= n_train

		print(f"Epoch: {epoch:02}")
		print(f"\tTrain Loss: {train_loss:.3f}")

		val_loss, val_acc, val_f1 = test(model, criterion, val_iter)
		print(f"\t Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc:.3f} |  Val. F1: {val_f1:.3f}")

		if val_acc > max_val_acc:
			max_val_acc = val_acc
			torch.save(model.state_dict(), BEST_MODEL_PATH)

		train_losses.append(train_loss)
		val_losses.append(val_loss)

	return train_losses, val_losses


def test(model, criterion, iterator, to_print=False):
	true_list = []
	pred_list = []
	n_test = len(iterator)

	test_loss = 0.0

	model.eval()
	with torch.no_grad():
		for batch in iterator:
			labels = batch.answer

			logits = model(batch)
			loss = criterion(logits, labels)
			test_loss += loss.item()

			logits = logits.detach()
			probs = torch.sigmoid(logits)
			preds = (probs > 0.5).type(torch.long).cpu()

			true_list.append(labels.detach().cpu())
			pred_list.append(preds)

	test_loss /= n_test

	true_labels = torch.cat(true_list, dim=0)
	pred_labels = torch.cat(pred_list, dim=0)

	test_acc = accuracy_score(true_labels, pred_labels)
	test_f1 = f1_score(true_labels, pred_labels)

	if to_print:
		print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | Test F1: {test_f1:.3f}")

	return test_loss, test_acc, test_f1


if __name__ == "__main__":
	from torchtext.vocab import GloVe

	vectors = GloVe(name="6B", dim=300)
	loader = Loader(32, 64, vectors, DEVICE)

	config = dict(
		pretrained=loader.WORD.vocab.vectors,
		char_vocab_size=len(loader.CHAR.vocab),
		char_emb_dim=15,
		char_hidden_size=15,
		char_kernel_size=5,
		emb_dim=300,
		hidden_size=100,
		dropout=0.2)

	set_seed()
	model = BiDAF(**config).to(DEVICE)

	n_epochs = 10
	clip = 3

	# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5)
	optimizer = torch.optim.Adam(model.parameters())
	criterion = nn.BCEWithLogitsLoss()

	print("training...")
	history = train_loop(
		model,
		optimizer,
		criterion,
		loader.train_iter,
		loader.val_iter,
		n_epochs,
		clip)

	test(model, criterion, loader.test_iter, to_print=True)

	best_model = BiDAF(**config).to(DEVICE)
	best_model.load_state_dict(torch.load(BEST_MODEL_PATH))

	test(best_model, criterion, loader.test_iter, to_print=True)

	# plot_training(*history)