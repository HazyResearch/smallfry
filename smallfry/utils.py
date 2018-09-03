import numpy as np
import marisa_trie as marisa


def load_embeddings(embeds_txt_filepath):
    """
    Loads a GloVe embedding at 'filename'. Returns a vector of strings that 
    represents the vocabulary and a 2-D numpy matrix that is the embeddings. 
    """
    with open(embeds_txt_filepath, 'r') as embeds_file:
        lines = embeds_file.readlines()
        wordlist = []
        embeddings = []
        for line in lines:
            print(line)
            row = line.strip("\n").split(" ")
            wordlist.append(row.pop(0))
            embeddings.append([float(i) for i in row])
        embeddings = np.array(embeddings)
        wordtrie = marisa.Trie(wordlist)
        trie_order_embeds = np.zeros(embeddings.shape)
        for i in range(len(wordlist)):
            i_prime = wordtrie[wordlist[i]]
            trie_order_embeds[i_prime,:] = embeddings[i,:]
    return wordtrie, trie_order_embeds
