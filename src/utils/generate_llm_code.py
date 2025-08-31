import re, os, json, time
from dataclasses import dataclass
from typing import List
import openai
from code_requirements_extraction import Spec

def build_prompt(spec: Spec):
    # prompt template
    PROMPT = """You are a precise Python code generator.  
        Implement code that satisfies all of the following requirements:  
        {reqs}  

        Guidelines:  
        - The code may include functions, classes, or a full module, depending on the requirements.  
        - The implementation must be clean, correct, and directly executable.
        - Do not include explanations, comments, or extra text.  
        - Only return the Python code in a proper code block. 
    """
    # fill in the prompt
    reqs = "\n".join([f"{i}. {r}" for i, r in enumerate(spec.requirements, 1)])
    return PROMPT.format(reqs=reqs)

def call_llm(model: str, prompt: str, temperature: float=0.0):
    # determine provider from model string
    provider = model.split(":")[0]

    # strip provider prefix
    model = model.split(":")[1]

    # call appropriate LLM provider
    if (provider == "openai"):        
        # openai api key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # check if api key exists
        if not openai.api_key:
            raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
        
        # call OpenAI
        response = openai.chat.completions.create(
            model=model,  
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        # return the output text
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")

def extract_code(text: str):
    # regex to extract code block
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    # if no code block, return the whole text stripped
    return m.group(1).strip() if m else text.strip()

def gen_candidates(spec: Spec, models: List[str], temps=[0.0,0.7], k=2):
    # make candidates and cache dirs
    os.makedirs("candidates", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    # cache path
    cache_path = f"cache/{spec.snippet_id}.jsonl"

    # seen contains keys of already used model,temp,k combos
    seen = set()

    # out contains list of output file paths
    outs = []

    # loop over models, temps, k
    for m in models:
        for t in temps:
            for i in range(k):
                # construct unique key of model,temp,k combo
                key = f"{m}:{t}:{i}"

                # if already seen, skip
                if key in seen: continue

                # call llm and get output code
                raw = call_llm(m, build_prompt(spec), temperature=t)
                code = extract_code(raw or "")

                # consruct unique filename
                fname = f"candidates/{spec.snippet_id}__{m}__t{t}__k{i}.py"

                # save code and info
                with open(fname, "w") as f: f.write(code)
                with open(cache_path, "a") as f: f.write(json.dumps({"key":key,"path":fname})+"\n")

                # add to outs and seen
                outs.append(fname)
                seen.add(key)

                # sleep to avoid rate limit
                time.sleep(0.2)
    return outs
