# src/features/build_reverseng.py

import os, json
import pandas as pd
from datasets import load_dataset
from src.utils.code_requirements_extraction import analyze_source, save_spec, Spec
from src.utils.generate_llm_code import gen_candidates_with_thread_pool
from src.utils.code_comparison import compare_against_candidates

DATASET_NAME = "serafeimdossas/ai-code-detection"

# folder structure
SPECS_FOLDER = "artifacts/specs"
CANDIDATES_FOLDER = "artifacts/candidates"
CACHE_FOLDER = "artifacts/cache"
ORIGINAL_FOLDER = "artifacts/original"
DATASET_FOLDER = "artifacts/dataset"

# LLMs configuration
# models list e.g., "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet", "vertex:codegemini"
# LLM_MODELS = ["openai:gpt-4o-mini", "google:gemini-2.5-flash-lite", "anthropic:claude-3-5-haiku-20241022"]
LLM_MODELS = ["openai:gpt-4o-mini", "anthropic:claude-3-5-haiku-20241022"]
# TEMPS = [0.0, 0.7] # deterministic + diverse
TEMPS = [0.7] # deterministic + diverse
K = 3 # candidates per (model, temp)

# weights for fused value calculation
WEIGHTS = {"lev": 0.15, "tok_jacc": 0.15, "ast": 0.25, "sem": 0.45}

# Loads samples from the specified dataset
def load_samples_from_dataset(number_of_samples=None):
    # load dataset
    dataset = load_dataset(DATASET_NAME)
    samples = dataset["train"]
    # check if subset was requested
    if number_of_samples is not None:        
        # keep first `number_of_samples` samples
        samples = dataset["train"].select(range(number_of_samples)) # type: ignore
    print(f"Loaded {len(samples)} samples") # type: ignore
    return samples

# Extracts code requirements from dataset samples
def extract_code_requirements(samples):        
    # Ensure original directory exists
    os.makedirs(ORIGINAL_FOLDER, exist_ok=True)

    # array to store occured errors for future adjustments 
    errors = []

    # array for storing samples with no requirements extracted
    empty = []

    # array for storing snippet_id and their labels
    result = []

    # keep count of spec files created
    count = 0
    
    # parse dataset samples
    for i, sample in enumerate(samples):
        # print statement for track of progress
        if i % 10 == 0:
            print(f"Processing sample {i}/{len(samples)}")
        
        # get current code and its label
        code = sample["code"]
        label = sample["label"]
        
        try:
            # analyze code of sample
            requirements = analyze_source(code)
        except Exception as e:
            # error occured, mone on to next sample
            errors.append(str(e))
            # print(f"Error analyzing sample {sample['task_name']}: {e}")
            continue
        
        # save spec file only when requirements of code were extracted
        if len(requirements.requirements) > 0:
            # save spec file
            save_spec(requirements, SPECS_FOLDER)

            # save original code
            original_code_path = f"{ORIGINAL_FOLDER}/{requirements.snippet_id}.py"
            with open(original_code_path, "w", encoding="utf-8") as f:
                f.write(code)

            # update result array and counter
            result.append({"snippet_id": requirements.snippet_id, "label": label})
            count += 1
        else:
            # keep track of samples with no requirements extracted
            empty.append(sample["task_name"])

    # print summary of process
    print(f"Finished processing {len(samples)} samples with {len(errors)} errors.")
    print(f"Samples with no requirements extracted: {len(empty)}")
    print(f"Total spec files created: {count}")
    
    # return array of ids for codes successfully analyzed along with their labels
    return result

def generate_ai_code():
    # find specs folder and files
    spec_files = sorted([f for f in os.listdir(SPECS_FOLDER) if f.endswith(".json")])
    
    # parse found spec files
    for spec_file in spec_files:
        # open spec file and get its content
        spec_path = os.path.join(SPECS_FOLDER, spec_file)
        with open(spec_path, "r") as f:
            spec_data = json.load(f)

        # construct spec object
        spec = Spec(**spec_data)

        # generate candidates for current spec file
        gen_candidates_with_thread_pool(spec, LLM_MODELS, CANDIDATES_FOLDER, CACHE_FOLDER, TEMPS, K)

def calculate_fused_values(snippet_ids):
    # parse array
    for i, item in enumerate(snippet_ids):
        # construct original and candidates file names for current snippet id
        candidates_paths = [f"{CANDIDATES_FOLDER}/{item["snippet_id"]}__{model}__t{temp}__k{k}.py" for model in LLM_MODELS for temp in TEMPS for k in range(K)]
        original_path = f"{ORIGINAL_FOLDER}/{item["snippet_id"]}.py"

        # compare original code to candidates
        per_candidate = compare_against_candidates(original_path, candidates_paths)

        # calculate fused value
        for pc in per_candidate:
            fused_value = sum(pc["scores"][metric] * WEIGHTS[metric] for metric in WEIGHTS)
            pc["fused"] = fused_value

        # get max fused value
        fused_values = [pc["fused"] for pc in per_candidate]
        max_fused = max(fused_values) if fused_values else 0.0

        # update array to include max_fused value
        snippet_ids[i]["max_fused"] = max_fused

def create_csv_dataset(data):
    # ensure dataset folder exists
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # create csv file path
    csv_path = os.path.join(DATASET_FOLDER, "code_fused_dataset.csv")

    # create dataframe and save as csv
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"CSV dataset created at {csv_path}")

def main():
    # get code samples from dataset
    number_of_samples = 200  # set to None to load all samples
    code_samples = load_samples_from_dataset(number_of_samples)

    # extract code requirements from dataset code samples and save as spec files
    # response contains snippet ids and their labels for codes successfully analyzed
    snippet_ids = extract_code_requirements(code_samples)

    # generate candidates codes for the created spec files
    generate_ai_code()

    # calculate max fused value for analyzed codes and add to existing array
    calculate_fused_values(snippet_ids)

    # create csv dataset from final results
    create_csv_dataset(snippet_ids)

if __name__ == "__main__":
    main()

