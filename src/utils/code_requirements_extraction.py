import ast, re, json, hashlib, os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

OUTPUT_FOLDER = "artifacts/specs"

# regex for detecting comments that may indicate requirements
COMMENT_HINTS = re.compile(r"\b(shall|must|should|required|todo|fixme)\b", re.I)

# keywords indicating certain actions e.g. I/O, HTTP, DB, logging
HTTP_FUNCS = {("requests", f) for f in ["get","post","put","patch","delete","head","options"]}
FILE_FUNCS = {("builtins","open")}
PRINT_FUNCS = {("builtins","print")}
DB_MODULES = {"sqlite3", "psycopg2", "mysql", "sqlalchemy"}
LOG_FUNCS = {("logging", f) for f in ["debug","info","warning","error","critical","exception"]}# Define a dataclass for the specification

@dataclass
class StaticInfo:
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    io: List[str]
    http: List[str]
    db: List[str]
    exceptions: List[str]
    cli: bool
    logging: List[str]

@dataclass
class Spec:
    snippet_id: str
    requirements: List[str]
    static_info: StaticInfo

# create unique hash for code snippet
def hash_code_snippet(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()[:12]

# get comments from source code
def get_comments(src: str):
    lines = []
    # parse lines for comments
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # remove leading # and keep rest
            lines.append(stripped[1:].strip())
    return lines

# define analyzer class, inheriting from ast.NodeVisitor
class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports: Dict[str, str] = {}           # name -> module
        self.from_imports: Dict[str, str] = {}      # alias -> module
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.raises: List[str] = []
        self.argparse_used = False
        self.sys_argv_used = False
        self.io_ops: List[str] = []                # "file read", "file write"
        self.http_ops: List[str] = []              # "HTTP GET to ..."
        self.db_ops: List[str] = []                # "DB connect/execute"
        self.logging_ops: List[str] = []
        self.print_used = False
        self.returns: List[str] = []
        self.docstrings: List[str] = []

    # --- imports methods ---
    def visit_Import(self, node: ast.Import):
        # loop through imported names
        for alias in node.names:
            # save as name -> module
            self.imports[alias.asname or alias.name] = alias.name
        # continue walking
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # get module name
        mod = node.module or ""
        for alias in node.names:
            # save as name -> module
            self.from_imports[alias.asname or alias.name] = mod
        # continue walking
        self.generic_visit(node)

    # --- defs and class methods ---
    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        # get args of def
        args = [a.arg for a in node.args.args]
        
        # get docstring if any and save
        doc = ast.get_docstring(node) or ""        
        if doc:
            self.docstrings.append(doc)

        # get return annotation if any   
        returns = ast.unparse(node.returns) if node.returns is not None else None

        # save function info and continue walking
        self.functions.append({"name": node.name, "args": args, "returns": returns, "doc": doc})
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # use same processing as for regular functions
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # get docstring if any and save
        doc = ast.get_docstring(node) or ""
        if doc:
            self.docstrings.append(doc)

        # get names of methods in class
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        # save class info and continue walking
        self.classes.append({"name": node.name, "methods": methods, "doc": doc})
        self.generic_visit(node)

    # --- methods for calls and attributes ---

    # return (root_name, attr_name) for Call func like logging.info / requests.get / open
    def _qual(self, node: ast.AST) -> Optional[tuple]:
        # handle simple names
        if isinstance(node, ast.Name):
            return (node.id, "")
        
        # handle attribute chains
        if isinstance(node, ast.Attribute):
            base = node
            parts = []
            # loop bakwards through attributes
            while isinstance(base, ast.Attribute):
                parts.append(base.attr)
                base = base.value

            # final base should be a Name
            if isinstance(base, ast.Name):
                parts.append(base.id)
                parts = list(reversed(parts))
                root = parts[0]
                attr = parts[1] if len(parts) > 1 else ""
                return (root, attr)
        
        # return None if not recognized
        return None

    def visit_Call(self, node: ast.Call):
        # get qualified name of function being called
        qual = self._qual(node.func)

        if qual:
            # get parts of qualified name
            root, attr = qual

            # map root through imports
            module = self.imports.get(root) or self.from_imports.get(root) or root

            # argparse / CLI
            if module == "argparse":
                self.argparse_used = True
            if module == "sys" and attr == "argv":
                self.sys_argv_used = True

            # print
            if (module, (attr or root)) in PRINT_FUNCS or (root == "print"):
                self.print_used = True

            # logging
            if (module, attr) in LOG_FUNCS:
                self.logging_ops.append(attr)

            # HTTP
            if (module, attr) in HTTP_FUNCS:
                url = None
                if node.args:
                    try:
                        url = ast.literal_eval(node.args[0])
                    except Exception:
                        url = None
                self.http_ops.append(f"{attr.upper()}{' ' + url if url else ''}".strip())

            # File I/O
            if (module, attr or root) in {("builtins","open")} or (root == "open"):
                # detect mode
                mode = "r"
                if len(node.args) >= 2:
                    try:
                        mode = ast.literal_eval(node.args[1])
                    except Exception:
                        pass
                elif any(k.arg == "mode" for k in node.keywords):
                    for k in node.keywords:
                        if k.arg == "mode":
                            try:
                                mode = ast.literal_eval(k.value)
                            except Exception:
                                pass
                if "w" in str(mode) or "a" in str(mode) or "+" in str(mode):
                    self.io_ops.append("file write")
                else:
                    self.io_ops.append("file read")

            # DB
            if module in DB_MODULES:
                self.db_ops.append(f"db via {module}")

        # continue walking
        self.generic_visit(node)

    # method for raises
    def visit_Raise(self, node: ast.Raise):
        # get exception type
        if node.exc is not None:
            try:
                # attempt to unparse the exception type
                name = ast.unparse(node.exc)
            except Exception:
                # fallback if unparsing fails
                name = "Exception"
            # store the exception type
            self.raises.append(name)
        # continue walking
        self.generic_visit(node)

    # method for return statements 
    def visit_Return(self, node: ast.Return):
        try:
            # unparse the return expression
            expr = ast.unparse(node.value) if node.value is not None else "None"
        except Exception:
            # fallback if unparsing fails
            expr = "value"
        # store the return expression
        self.returns.append(expr)
        # continue walking
        self.generic_visit(node)

# turn string into sentence case
def sentence_case(string: str):
    string = string.strip()
    return string[0:1].upper() + string[1:] if string else string

# create list of functional requirements from analysis
def to_requirements(analysis: Analyzer, comments: List[str], code: str):
    reqs: List[str] = []
    details: Dict[str, Any] = {}

    # From functions
    for f in analysis.functions:
        if f["args"]:
            reqs.append(f"The system shall provide a function `{f['name']}` that accepts parameters: {', '.join(f['args'])}.")
        else:
            reqs.append(f"The system shall provide a function `{f['name']}` with no required parameters.")
        if f["returns"]:
            reqs.append(f"The function `{f['name']}` shall return `{f['returns']}`.")
        if f["doc"]:
            reqs.append(f"The function `{f['name']}` shall fulfill: {sentence_case(f['doc'].splitlines()[0])}")

    # From classes
    for c in analysis.classes:
        reqs.append(f"The system shall define a class `{c['name']}` with methods: {', '.join(c['methods']) or 'none'}.")
        if c["doc"]:
            reqs.append(f"The class `{c['name']}` shall: {sentence_case(c['doc'].splitlines()[0])}")

    # Side effects
    for op in sorted(set(analysis.io_ops)):
        if op == "file read":
            reqs.append("The system shall read data from files.")
        if op == "file write":
            reqs.append("The system shall persist output to files.")

    for http in sorted(set(analysis.http_ops)):
        verb, _, url = http.partition(" ")
        if url:
            reqs.append(f"The system shall perform HTTP {verb} requests to {url}.")
        else:
            reqs.append(f"The system shall perform HTTP {verb} requests.")

    for dbo in sorted(set(analysis.db_ops)):
        reqs.append(f"The system shall interact with a database ({dbo}).")

    if analysis.print_used:
        reqs.append("The system shall display output to the console.")

    if analysis.logging_ops:
        reqs.append("The system shall record runtime events using the logging library.")

    # CLI
    if analysis.argparse_used or analysis.sys_argv_used:
        reqs.append("The system shall provide a command-line interface for user input.")

    # Exceptions
    if analysis.raises:
        kinds = ", ".join(sorted(set(analysis.raises)))
        reqs.append(f"The system shall raise appropriate exceptions ({kinds}) upon invalid states or inputs.")

    # Comment hints
    hinted = [c for c in comments if COMMENT_HINTS.search(c)]
    for c in hinted:
        reqs.append(f"[Comment hint] {sentence_case(c)}")

    # Minimal dedup / normalization
    reqs = sorted(set(reqs), key=reqs.index)

    details["functions"] = analysis.functions
    details["classes"] = analysis.classes
    details["io"] = analysis.io_ops
    details["http"] = analysis.http_ops
    details["db"] = analysis.db_ops
    details["exceptions"] = analysis.raises
    details["cli"] = analysis.argparse_used or analysis.sys_argv_used
    details["logging"] = analysis.logging_ops

    return Spec(
        snippet_id=hash_code_snippet(code),
        requirements=reqs,
        static_info=StaticInfo(
            functions=analysis.functions,
            classes=analysis.classes,
            io=analysis.io_ops,
            http=analysis.http_ops,
            db=analysis.db_ops,
            exceptions=analysis.raises,
            cli=analysis.argparse_used or analysis.sys_argv_used,
            logging=analysis.logging_ops
        )
    )

# analyze code snippet and return extracted requirements
def analyze_source(src: str):
    # get AST of code
    tree = ast.parse(src)
    
    # create Analyzer object and visit AST
    analyzer = Analyzer()
    analyzer.visit(tree)

    # get comments from code if any
    comments = get_comments(src)

    # get and return extracted requirements
    return to_requirements(analyzer, comments, src)

def save_spec(spec: Spec, folder):
    # Ensure the output directory exists
    os.makedirs(folder, exist_ok=True)
    # Save the spec as a JSON file
    with open(os.path.join(folder, f"{spec.snippet_id}.json"), "w") as f:
        json.dump(asdict(spec), f, indent=2)

def main():
    code = """def factorial(n):
    # Calculate factorial of n
    # The input n shall be a non-negative integer
    # if zero, return 1
    if n < 0:
        raise ValueError("Negative not allowed")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n+1):
         result *= i
    return result
    """

    # Analyze source code
    result = analyze_source(code)

    # Save code spec to file
    save_spec(result, OUTPUT_FOLDER)

    # Pretty print
    print("=== Functional Requirements ===")
    print("\n".join([f"{i}. {r}" for i, r in enumerate(result.requirements, 1)]))

if __name__ == "__main__":
    main()
