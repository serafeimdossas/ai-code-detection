import re, os, json, time
from dataclasses import dataclass
from typing import List
import openai
from anthropic import Anthropic
import google.generativeai as genai
from src.utils.code_requirements_extraction import Spec
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_prompt(spec: Spec):
    # reqs array for filling the prompt template
    reqs = "\n".join([f"{i}. {r}" for i, r in enumerate(spec.requirements, 1)])
    # prompt template
    return (
        "You are a precise Python code generator.\n"
        "Implement code that satisfies all of the following requirements:\n"
        f"{reqs}\n\n"
        "Guidelines:\n"
        "- The code may include functions, classes, or a full module, depending on the requirements.\n"
        "- The implementation must be clean, correct, and directly executable.\n"
        "- Do not include explanations, comments, or extra text.\n"
        "- Only return the Python code in a proper code block.\n"
    )

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
    elif (provider == "anthropic"):
        # anthropic api key
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # check if api key exists
        if not anthropic_api_key:
            raise RuntimeError("Please set the ANTHROPIC_API_KEY environment variable.")
        
        # create anthropic client
        client = Anthropic(api_key=anthropic_api_key)

        # call Anthropic
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # return the output text
        return response.content[0].text # type: ignore
    elif (provider == "google"):
        # gemini api key
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        # check if api key exists
        if not gemini_api_key:
            raise RuntimeError("Please set the GEMINI_API_KEY environment variable.")
        
        # set api key
        genai.configure(api_key=gemini_api_key) # type: ignore

        # Initialize the Gemini model
        gemini_model = genai.GenerativeModel(model)  # type: ignore

        # call Google Gemini API
        response = gemini_model.generate_content(prompt)

        # return the output text
        return response.text
    else:
        raise ValueError(f"Unknown provider: {provider}")

def extract_code(text: str):
    # regex to extract code block
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    # if no code block, return the whole text stripped
    return m.group(1).strip() if m else text.strip()

def gen_candidates(spec: Spec, models: List[str], candidates_folder: str, cache_folder: str, temps=[0.0,0.7], k=2):
    # make candidates and cache dirs
    os.makedirs(candidates_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)

    # cache path
    cache_path = f"{cache_folder}/{spec.snippet_id}.jsonl"

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
                fname = f"{candidates_folder}/{spec.snippet_id}__{m}__t{t}__k{i}.py"

                # save code and info
                with open(fname, "w") as f: f.write(code)
                with open(cache_path, "a") as f: f.write(json.dumps({"key":key,"path":fname})+"\n")

                # add to outs and seen
                outs.append(fname)
                seen.add(key)

                # sleep to avoid rate limit
                time.sleep(0.2)
    return outs

def gen_candidates_with_thread_pool(spec: Spec, models: List[str], candidates_folder: str, cache_folder: str, temps=[0.0,0.7], k=2):
    # make candidates and cache dirs
    os.makedirs(candidates_folder, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)

    # cache path
    cache_path = f"{cache_folder}/{spec.snippet_id}.jsonl"

    # seen contains keys of already used model,temp,k combos
    seen = set()

    # out contains list of output file paths
    outs = []

    # function to call llm and save code
    def call_and_save(m, t, i):
        # construct unique key of model,temp,k combo
        key = f"{m}:{t}:{i}"

        # if already seen, skip
        if key in seen: return None

        # call llm and get output code
        raw = call_llm(m, build_prompt(spec), temperature=t)
        code = extract_code(raw or "")

        # consruct unique filename
        fname = f"{candidates_folder}/{spec.snippet_id}__{m}__t{t}__k{i}.py"

        # save code and info
        with open(fname, "w") as f: f.write(code)
        with open(cache_path, "a") as f: f.write(json.dumps({"key":key,"path":fname})+"\n")

        # add to seen
        seen.add(key)

        return fname

    # use ThreadPoolExecutor to parallelize calls
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for m in models:
            for t in temps:
                for i in range(k):
                    futures.append(executor.submit(call_and_save, m, t, i))

        for future in as_completed(futures):
            result = future.result()
            if result:
                outs.append(result)
                # sleep to avoid rate limit
                time.sleep(0.2)

    return outs