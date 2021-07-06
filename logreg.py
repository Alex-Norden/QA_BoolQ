import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from transformers import DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


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


MODEL_CLASSES = {
	"distilbert-base-uncased": (DistilBertModel, DistilBertTokenizer),
	"bert-base-uncased": (BertModel, BertTokenizer),
	"bert-large-uncased": (BertModel, BertTokenizer),
	"roberta-base": (RobertaModel, RobertaTokenizer),
	"roberta-large": (RobertaModel, RobertaTokenizer),
}


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


class TokensDataset(Dataset):
	def __init__(self, column, tokenizer, max_length):
		self.tokenized = column.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_length)))

	def __getitem__(self, idx):
		return self.tokenized[idx]

	def __len__(self):
		return len(self.tokenized)


def get_padded(values):
	max_len = 0
	for value in values:
		if len(value) > max_len:
			max_len = len(value)

	padded = np.array([value + [0]*(max_len-len(value)) for value in values])
	return padded

def get_dataloader(column, tokenizer, max_length, batch_size):
	def collate_fn(batch_ids):
		input_ids = get_padded(batch_ids) #padded input_ids
		attention_mask = np.where(input_ids != 0, 1, 0)
		return torch.tensor(input_ids), torch.tensor(attention_mask)

	dataset = TokensDataset(column, tokenizer, max_length)
	return DataLoader(dataset, collate_fn=collate_fn)


def run_model(model_name, batch_size=32, max_length=256, grid_search=False):
	def get_cat_embeddings(df):
		def get_embeddings(column):
			loader = get_dataloader(column, tokenizer, max_length, batch_size)

			embeddings = []

			with torch.no_grad():
				for batch in loader:
					input_ids = batch[0].to(DEVICE)
					attention_mask = batch[1].to(DEVICE)

					last_hidden_states = model(input_ids, attention_mask)
					batch_emb = last_hidden_states[0].detach().cpu()
					embeddings.append(batch_emb.mean(dim=1))

			return torch.cat(embeddings, dim=0)

		q_embs = get_embeddings(df.question)
		p_embs = get_embeddings(df.passage)
		return torch.cat([q_embs, p_embs], dim=-1).numpy()

	def run_grid_search():
		param_grid = {
			"C": (0.001, 0.01, 0.1, 1, 10, 100),
			"penalty": ("l1", "l2"),
		}

		# kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
		kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

		estimator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED)

		grid_cv = GridSearchCV(estimator=estimator,
								param_grid=param_grid,
								cv=kf,
								scoring="roc_auc",
								n_jobs=-1,
								verbose=10,
								refit=True)
		grid_cv.fit(train_features, train_labels)

		print("best_params:", grid_cv.best_params_)
		print("best_score:", grid_cv.best_score_)
		print("best_estimator:", grid_cv.best_estimator_)

		return grid_cv.best_estimator_

	# load pretrained model/tokenizer
	model_class, tokenizer_class = MODEL_CLASSES[model_name]

	tokenizer = tokenizer_class.from_pretrained(model_name)
	model = model_class.from_pretrained(model_name).to(DEVICE)

	set_seed()

	print("embeddings...")
	train_features = get_cat_embeddings(df_train)
	test_features = get_cat_embeddings(df_dev)

	train_labels = df_train.answer.values
	test_labels = df_dev.answer.values

	print("training...")
	if grid_search:
		print("grid search...")
		lr_clf = run_grid_search()
	else:
		# lr_clf = LogisticRegression()
		lr_clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=0.1, penalty="l2", random_state=SEED)
		lr_clf.fit(train_features, train_labels)

	print("prediction...")
	pred_labels = lr_clf.predict(test_features)
	test_acc = accuracy_score(test_labels, pred_labels)
	test_f1 = f1_score(test_labels, pred_labels)
	print(f"Test Acc: {test_acc:.3f} | Test F1: {test_f1:.3f}")
	return test_acc, test_f1


if __name__ == "__main__":
	run_model("distilbert-base-uncased")