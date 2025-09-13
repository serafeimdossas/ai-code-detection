# src/features/build_reverseng.py

import os, json
from datasets import load_dataset
from src.utils.code_requirements_extraction import analyze_source, save_spec, Spec
from src.utils.generate_llm_code import gen_candidates

DATASET_NAME = "serafeimdossas/ai-code-detection"

# folder structure
SPECS_FOLDER = "artifacts/specs"
CANDIDATES_FOLDER = "artifacts/candidates"
CACHE_FOLDER = "artifacts/cache"

# LLMs configuration
LLM_MODELS = ["openai:gpt-4o-mini"] # e.g., "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet", "vertex:codegemini"
TEMPS = [0.0, 0.7]                  # deterministic + diverse
K = 2                               # candidates per (model, temp)

# Loads samples from the specified dataset
def load_samples_from_dataset(number_of_samples):
    # load dataset
    dataset = load_dataset(DATASET_NAME)
    # keep first `number_of_samples` samples
    samples = dataset["train"].select(range(number_of_samples)) # type: ignore
    return samples

# Extracts code requirements from dataset samples
def extract_code_requirements(samples):
    # array to store occured errors for future adjustments 
    errors = []

    # keep count of spec files created
    count = 0
    
    # parse dataset samples
    for i, sample in enumerate(samples):
        # print statement for track of progress
        if i % 10 == 0:
            print(f"Processing sample {i}/{len(samples)}")
        
        try:
            # analyze code of sample
            code = sample["code"]
            requirements = analyze_source(code)
        except Exception as e:
            # error occured, mone on to next sample
            errors.append(str(e))
            # print(f"Error analyzing sample {sample['task_name']}: {e}")
            continue

        # print(f"{sample['task_name']} - {requirements.snippet_id}")
        
        # save spec file only when requirements of code were extracted
        if len(requirements.requirements) > 0:
            save_spec(requirements, SPECS_FOLDER)
            count += 1

    print(f"Finished processing {len(samples)} samples with {len(errors)} errors.")
    print(f"Total spec files created: {count}")

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
        cand_paths = gen_candidates(spec, LLM_MODELS, CANDIDATES_FOLDER, CACHE_FOLDER, TEMPS, K)

        # experimental break of loop
        break


def main():
    # extract code requirements from dataset samples and save as spec files
    number_of_samples = 10
    code_samples = load_samples_from_dataset(number_of_samples)
    extract_code_requirements(code_samples)

    # generate candidates spec files created
    generate_ai_code()

if __name__ == "__main__":
    main()

