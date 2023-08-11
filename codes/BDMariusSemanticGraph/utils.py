from nltk import Tree

def buildTree(token):
    if token.n_lefts + token.n_rights > 0:
        return Tree(token, [buildTree(child) for child in token.children])
    else:
        return buildTree(token)