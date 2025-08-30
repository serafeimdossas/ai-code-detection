import json, os, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import ast, asttokens

OUTPUT_DIR = "artifacts/specs"

# Define a dataclass to hold static analysis results
@dataclass
class StaticInfo:
    language: str
    functions: List[Dict[str, Any]]
    imports: List[str]
    has_io: bool
    globals_written: List[str]

# Define a dataclass for the specification
@dataclass
class Spec:
    snippet_id: str
    summary: str
    requirements: List[str]
    edge_cases: List[str]
    signature_hint: Dict[str, Any]
    constraints: List[str]
    tests_hint: List[Dict[str, Any]]

# create unique hash for code snippet
def hash_code_snippet(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()[:12]

def extract_static_python(code: str):
    # Parse the code into an AST
    atok = asttokens.ASTTokens(code, parse=True)
    
    # Get the AST tree
    tree = atok.tree or ast.Module(body=[], type_ignores=[])
    
    # Initialize data structures
    funcs, imps, writes = [], [], set()
    has_io = False

    # Walk through the AST nodes
    for node in ast.walk(tree):
        # get function definitions of code
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            returns = ast.get_source_segment(code, node.returns) if node.returns else None
            funcs.append({"name": node.name, "args": args, "returns": returns})
        
        # get imports of code
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", None)
            names = [n.name for n in node.names]
            imps.extend([mod] if mod else names)
        
        # check for I/O operations in code
        if isinstance(node, ast.Call):
            fn = ast.get_source_segment(code, node.func)
            if fn and any(k in fn for k in ["open(", "print(", "requests.", "os.", "sys."]):
                has_io = True

        # track global variable writes
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    writes.add(t.id)

    # return the collected static information
    return StaticInfo(language="python", functions=funcs, imports=imps, has_io=has_io, globals_written=list(writes))

def draft_spec_from_static(code: str, static_info: StaticInfo):
    # use first function found or a default
    fn = static_info.functions[0] if static_info.functions else {"name": "solution", "args": ["x"], "returns": None}

    # behavior summary and requirements based on static analysis
    summary = "Functionality inferred via AST: pure computation over inputs; no external I/O." if not static_info.has_io else "Function may perform I/O; constrain to pure interface for testing."
    
    # requirements list, includes default values
    reqs = [
        "Produce deterministic output for identical inputs.",
        "Handle empty or None-like inputs gracefully where applicable.",
        "Raise ValueError on clearly invalid inputs."
    ]

    # edge cases and constraints, including defaults
    edges = ["empty list / string", "very large input size", "invalid types"]
    constraints = ["No network calls", "No filesystem writes", "Deterministic behavior"]

    # example tests cases with placeholders
    tests_hint = [{"input":[1,2,3], "expect":"fill_me"}, {"input":[],"expect":"fill_me"}]

    # return the drafted specification
    return Spec(
        snippet_id=hash_code_snippet(code),
        summary=summary,
        requirements=reqs,
        edge_cases=edges,
        signature_hint=fn,
        constraints=constraints,
        tests_hint=tests_hint
    )

def save_spec(spec: Spec, folder):
    # Ensure the output directory exists
    os.makedirs(folder, exist_ok=True)
    # Save the spec as a JSON file
    with open(os.path.join(folder, f"{spec.snippet_id}.json"), "w") as f:
        json.dump(asdict(spec), f, indent=2)

if __name__ == "__main__":
    # example code snippet
    code = """def factorial(n):
    if n < 0:
        raise ValueError("Negative not allowed")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n+1):
         result *= i
    return result
    """

    # Step 1: Extract static info
    static_info = extract_static_python(code)
    print("Static Info:", static_info)

    # Step 2: Draft a spec from static info
    spec = draft_spec_from_static(code, static_info)
    print("\nDrafted Spec:", spec)

    # # Step 3: Save spec as JSON
    # save_spec(spec, OUTPUT_DIR)

    # # Step 4: Inspect the result
    # spec_path = os.path.join(OUTPUT_DIR, f"{spec.snippet_id}.json")
    # print(f"\nSaved Spec at: {spec_path}")
    # print(json.dumps(json.load(open(spec_path)), indent=2))