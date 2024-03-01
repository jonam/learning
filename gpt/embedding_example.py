import torch

# Update the vocabulary to include space and correct the vocab size
vocab = {
    ' ': 0, 'Q': 1, 'u': 2, 'i': 3, 'c': 4, 'k': 5, 
    'b': 6, 'r': 7, 'o': 8, 'w': 9, 'n': 10, 
    'f': 11, 'x': 12, 'j': 13, 'm': 14, 'p': 15, 
    'e': 16, 'd': 17, 'v': 18, 't': 19, 'h': 20, 
    'l': 21, 'a': 22, 'z': 23, 'y': 24, 'g': 25, '.': 26}
vocab_size = len(vocab)  # Updated unique characters including space and period
sentence = "Quick brown fox jumped over the lazy dog."

# Update tensors for character indices and positions
char_indices = [vocab[c] for c in sentence]
positions = list(range(len(sentence)))
print(positions)
n_embd = 5

# Re-initialize embeddings with updated vocab size
token_embedding = torch.nn.Embedding(vocab_size, n_embd)
position_embedding = torch.nn.Embedding(len(sentence), n_embd)

# Generate embeddings again
token_embeddings = token_embedding(torch.tensor(char_indices))
position_embeddings = position_embedding(torch.tensor(positions))

print(f"token_embeddings = {token_embeddings}")
print(f"position_embeddings = {position_embeddings}")
