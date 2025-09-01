import editdistance, tokenize, io, ast
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("microsoft/codebert-base")

# Tokenize Python code into tokens
def tokenize_py(code: str):
    toks=[]
    # loop through code tokens
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        # only keep names, numbers, strings, and operators
        if tok.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP):
            # append token string to list
            toks.append(tok.string)
    return toks

# calculate levenshtein similarity between two strings
def lev_similarity(a: str, b: str):
    # split code by whitespace separator
    La, Lb = a.split(), b.split()

    # compute normalized Levenshtein similarity
    ed = editdistance.eval(La, Lb)

    # normalize by length of longer snippet
    max_len = max(len(La), len(Lb), 1)
    sim = 1 - ed / max_len

    # keep values to [0,1]
    return max(0.0, min(1.0, sim))

# calculate lexical similarity scores between two code snippets
def lexical_scores(a: str, b: str):
    # compute normalized Levenshtein similarity
    lev_sim = lev_similarity(a, b)

    # get token lists
    ta, tb = tokenize_py(a), tokenize_py(b)

    # compute Jaccard token overlap
    overlap = len(set(ta) & set(tb)) / max(1, len(set(ta) | set(tb)))

    # return dictionary of scores
    return {"lev": lev_sim, "tok_jacc": overlap}

# get ast node type names from code
def ast_shape(code: str):
    try:
        # parse code into AST
        tree = ast.parse(code)
        # return list of AST node type names
        return [type(n).__name__ for n in ast.walk(tree)]
    except Exception:
        return ["PARSE_ERROR"]

# compute Jaccard similarity between AST node type sets of two code snippets
def ast_jaccard(a: str, b: str):
    # get AST node type sets
    A, B = set(ast_shape(a)), set(ast_shape(b))
    # return Jaccard similarity of AST node type sets
    return len(A & B) / max(1, len(A | B))

# embed code snippet into vector
def embed(code: str):
    return _model.encode([code])[0]

# calculate cosine similarity between two vectors
def cos(a, b): 
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# compare code snippet A against list of candidate code snippets
def compare_against_candidates(code_A_path: str, cand_paths: List[str]):
    # read code A and embed
    a = open(code_A_path).read()
    a_emb = embed(a)

    out=[]

    # loop through candidate paths
    for cp in cand_paths:
        # read candidate code
        b = open(cp).read()
        
        # compute scores
        scores = {}
        scores.update(lexical_scores(a, b))
        scores["ast"] = ast_jaccard(a,b)
        scores["sem"] = cos(a_emb, embed(b))

        # append candidate path and scores to output list
        out.append({"candidate": cp, "scores": scores})
    
    # return output list
    return out