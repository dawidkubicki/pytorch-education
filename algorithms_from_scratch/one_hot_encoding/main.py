import numpy as np

text = "There was no other sentence to pick as an example"

token_seq = str.split(text)
num_tokens = len(token_seq)

vocab = sorted(token_seq)
vocab_size = len(vocab)


one_hot = np.zeros((num_tokens, vocab_size))

for idx, token in enumerate(token_seq):
	one_hot[idx, vocab.index(token)] = 1

print(one_hot)

