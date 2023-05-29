# import spacy

# nlp = spacy.load("en_core_web_sm")

# doc = nlp("The quick brown fox jumps over the lazy dog.")

# dep_vectors = []
# for token in doc:
#     print(token.text, token.vector.shape)
#     dep_vector = token.vector
#     dep_vectors.append(dep_vector)
    
# import torch
# import torch.nn as nn

# # Define the embedding layer with 10 vocab size and 50 vector embeddings.
# embedding = nn.Embedding(10, 50)

# print(embedding(torch.LongTensor([0])))
# print(embedding(torch.LongTensor([0])))


# import spacy
# nlp = spacy.load("en_core_web_sm")
# piano_text = "Gus is learning piano"
# piano_doc = nlp(piano_text)
# for token in piano_doc:
#     print(
#         f"""
#         TOKEN: {token.text}
#         =====
#         {token.tag_ = }
#         {token.head.text = }
#         {token.dep_ = }
#         {token.dep = }"""
#    )


import spacy
import torch.nn as nn
import torch

nlp = spacy.load('en_core_web_sm')

def create_embedding(sentence, embedding_model):
    # Tokenize the sentence using spaCy
    doc = nlp(sentence)
    
    # Create a list to store the embeddings and dependency parsing information
    embeddings = []
    dep_info = []
    
    # Iterate over each token in the sentence
    for token in doc:
        # Get the word embedding for the token using nn.embedding
        word_embedding = embedding_model(torch.tensor([token.i]))
        embeddings.append(word_embedding)
        
        # Get the dependency parsing information for the token
        dep_info.append(token.dep_)
        
    # Concatenate the embeddings and dependency parsing information into a single vector
    embedding = torch.cat(embeddings)
    print(embedding.shape)
    dep_tensor = torch.tensor([nlp.vocab.strings[s] for s in dep_info])
    dep_tensor = dep_tensor.unsqueeze(1) # add a new dimension
    print(dep_tensor.shape)
    embedding = torch.cat([embedding, dep_tensor], dim=1)  # concatenate along the second dimension

    print(embedding.shape)
    return embedding


# Load a pre-trained word embedding model
embedding_model = nn.Embedding(10000, 5)

# Create an embedding for a sentence
sentence = "The quick."
embedding = create_embedding(sentence, embedding_model)

# Print the resulting embedding vector
print(embedding)