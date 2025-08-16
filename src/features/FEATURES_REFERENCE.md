# Engineered Code Features — Definitions

## A) Line & Formatting Stats
- **n_lines** — Total number of lines in the code snippet.  
- **avg_line_len** — Average number of characters per line.  
- **stdev_line_len** — Standard deviation of line lengths (measures variation in line length).  
- **max_line_len** — Number of characters in the longest line.  
- **blank_ratio** — Proportion of lines that are completely blank.  
- **trailing_ws_ratio** — Proportion of lines ending with trailing whitespace.

## B) Indentation Stats
- **avg_indent** — Average indentation depth (spaces/tabs before code on each line).  
- **max_indent** — Maximum indentation depth in the snippet.  
- **tab_leading_ratio** — Proportion of non-blank lines starting with a tab character.

## C) Identifier Pattern Features
- **avg_ident_len** — Average length of identifiers (variables, functions, classes).  
- **single_letter_ratio** — Proportion of identifiers that are a single character long.  
- **snake_ratio** — Proportion of identifiers using `snake_case`.  
- **camel_ratio** — Proportion of identifiers using `camelCase`.  
- **upper_ratio** — Proportion of identifiers in `UPPER_CASE` (constants).

## D) AST Shape / Structure Features
- **n_FunctionDef** — Count of function definitions (`FunctionDef` + `AsyncFunctionDef`).  
- **n_ClassDef** — Count of class definitions.  
- **n_If** — Count of `if` statements.  
- **n_For** — Count of `for` loops (`For` + `AsyncFor`).  
- **n_While** — Count of `while` loops.  
- **n_Try** — Count of `try` blocks.  
- **n_With** — Count of `with` statements (`With` + `AsyncWith`).  
- **n_Call** — Count of function calls.  
- **n_ListComp** — Count of list comprehensions.  
- **n_DictComp** — Count of dictionary comprehensions.  
- **n_Lambda** — Count of lambda expressions.  
- **avg_func_len** — Average function length in lines (based on AST line numbers).  
- **max_ast_depth** — Maximum nesting depth in the AST.

## E) Import Features
- **n_imports** — Number of `import ...` statements.  
- **n_from_imports** — Number of `from ... import ...` statements.  
- **n_unique_imports** — Number of unique top-level packages/modules imported.

## F) Per-kLoC Normalizations
For each feature starting with `n_` (except `n_lines`):
- **{feature_name}_per_kloc** — Value normalized per 1000 lines of code (kLoC), e.g.:  
  - `n_FunctionDef_per_kloc`  
  - `n_ClassDef_per_kloc`  
  - `n_If_per_kloc`  
  - `n_Call_per_kloc`
