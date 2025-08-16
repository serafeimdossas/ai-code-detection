import ast
import io
import re
import tokenize
from collections import Counter

SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
CAMEL_RE = re.compile(r"^[a-z]+[A-Za-z0-9]*$")
UPPER_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

def safe_parse(code: str):
    try:
        return ast.parse(code)
    except Exception:
        return None

def iter_tokens(code: str):
    # tokenize Python code into lexical tokens.
    try:
        yield from tokenize.generate_tokens(io.StringIO(code).readline)
    except Exception:
        return []

def line_stats(lines):
    """
    Compute statistics about lines of code:
    - number of lines, average length, stdev of line length
    - max line length, ratio of blank lines, ratio of trailing whitespace lines
    """
    lengths = [len(l) for l in lines] or [0]
    n = len(lines)
    avg = sum(lengths)/len(lengths)
    var = sum((x-avg)**2 for x in lengths)/len(lengths)
    return {
        "n_lines": n,
        "avg_line_len": avg,
        "stdev_line_len": var ** 0.5,
        "max_line_len": max(lengths, default=0),
        "blank_ratio": sum(1 for l in lines if not l.strip()) / max(n,1),
        "trailing_ws_ratio": sum(1 for l in lines if len(l) and l.endswith((" ", "\t"))) / max(n,1),
    }

def indent_stats(lines):
    """
    Compute statistics about indentation:
    - average and maximum indentation width
    - ratio of lines indented with tabs instead of spaces
    """
    indents = []
    tabs = 0
    for l in lines:
        if not l.strip(): # skip blank lines
            continue
        # count leading spaces and tabs
        prefix = len(l) - len(l.lstrip(" \t"))
        indents.append(prefix)
        if l.startswith("\t"):
            tabs += 1
    return {
        "avg_indent": (sum(indents)/len(indents)) if indents else 0.0,
        "max_indent": max(indents) if indents else 0,
        "tab_leading_ratio": tabs / max(sum(1 for l in lines if l.strip()), 1),
    }

def identifier_features(tree: ast.AST | None):
    """
    Extract features related to identifiers in the AST:
    - average identifier length
    - ratio of single-letter identifiers
    - ratio of snake_case, camelCase, and UPPER_CASE identifiers
    """
    names = []
    if not tree:
        # No AST available (invalid code)
        return {"avg_ident_len": 0.0, "single_letter_ratio": 0.0, "snake_ratio": 0.0, "camel_ratio": 0.0, "upper_ratio": 0.0}
    
    # collect all identifier names from the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # use the identifier name directly
            names.append(node.id)
        elif isinstance(node, ast.alias):
            # import aliases
            names.append(node.asname or node.name.split(".")[-1])

    n = len(names)
    if n == 0:
        return {"avg_ident_len": 0.0, "single_letter_ratio": 0.0, "snake_ratio": 0.0, "camel_ratio": 0.0, "upper_ratio": 0.0}

    # count different identifier styles
    snake = sum(1 for s in names if SNAKE_RE.match(s))
    camel = sum(1 for s in names if CAMEL_RE.match(s) and not SNAKE_RE.match(s))
    upper = sum(1 for s in names if UPPER_RE.match(s))
    single = sum(1 for s in names if len(s) == 1)
    avg_len = sum(len(s) for s in names)/n
    return {
        "avg_ident_len": avg_len,
        "single_letter_ratio": single/n,
        "snake_ratio": snake/n,
        "camel_ratio": camel/n,
        "upper_ratio": upper/n,
    }

def ast_shape_features(tree: ast.AST | None):
    """
    Extract structural features from the AST:
    - counts of functions, classes, loops, conditionals, etc.
    - average function length
    - maximum AST depth (nesting complexity)
    """
    cnt = Counter()
    func_lengths = []
    max_depth = 0

    # Recursive function to calculate maximum depth of the AST
    def depth(node, d=0):
        nonlocal max_depth
        max_depth = max(max_depth, d)
        for ch in ast.iter_child_nodes(node):
            depth(ch, d+1)

    if tree:
        # Walk the AST and collect counts
        for node in ast.walk(tree):
            cnt[type(node).__name__] += 1
            # calculate function lengths
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "body") and node.body:
                    start = getattr(node, "lineno", None)
                    end = getattr(node.body[-1], "end_lineno", getattr(node.body[-1], "lineno", start))
                    if start and end and end >= start:
                        func_lengths.append(end - start + 1)
        
        # calculate maximum depth of the AST
        depth(tree)

    def g(name): return float(cnt.get(name, 0))
    avg_func_len = (sum(func_lengths)/len(func_lengths)) if func_lengths else 0.0
    return {
        "n_FunctionDef": g("FunctionDef") + g("AsyncFunctionDef"),
        "n_ClassDef": g("ClassDef"),
        "n_If": g("If"),
        "n_For": g("For") + g("AsyncFor"),
        "n_While": g("While"),
        "n_Try": g("Try"),
        "n_With": g("With") + g("AsyncWith"),
        "n_Call": g("Call"),
        "n_ListComp": g("ListComp"),
        "n_DictComp": g("DictComp"),
        "n_Lambda": g("Lambda"),
        "avg_func_len": avg_func_len,
        "max_ast_depth": float(max_depth),
    }

def import_features(tree: ast.AST | None):
    """
    Extract features related to imports:
    - number of plain `import` statements
    - number of `from ... import ...` statements
    - number of unique imported packages
    """
    if not tree:
        return {"n_imports": 0.0, "n_from_imports": 0.0, "n_unique_imports": 0.0}
    n_imp = n_from = 0
    pkgs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            n_imp += 1
            for a in node.names:
                pkgs.add((a.name or "").split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            n_from += 1
            pkgs.add((node.module or "").split(".")[0])
    return {
        "n_imports": float(n_imp),
        "n_from_imports": float(n_from),
        "n_unique_imports": float(len(pkgs)),
    }

def python_code_features(code: str) -> dict:
    """
    Main function to extract all features from Python code:
    - line-level stats
    - indentation stats
    - identifier naming features
    - AST structural features
    - import-related features
    - normalization per kLoC
    """

    # normalize line endings and split into lines
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")
    tree = safe_parse(code)

    feats = {}
    # Extract features from the code
    feats.update(line_stats(lines))
    feats.update(indent_stats(lines))
    feats.update(identifier_features(tree))
    feats.update(ast_shape_features(tree))
    feats.update(import_features(tree))

    # Normalizations (per kLoC)
    kloc = max(feats["n_lines"], 1) / 1000.0
    for k in list(feats.keys()):
        if k.startswith(("n_",)) and k not in ("n_lines",):
            feats[k + "_per_kloc"] = feats[k] / kloc
    return feats
