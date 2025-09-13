# src/features/build_reverseng.py

from datasets import load_dataset
from src.utils.code_requirements_extraction import analyze_source, save_spec

DATASET_NAME = "serafeimdossas/ai-code-detection"
SPECS_FOLDER = "artifacts/specs"

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

def main():
    # extract code requirements from dataset samples and save as spec files
    number_of_samples = 10
    code_samples = load_samples_from_dataset(number_of_samples)
    extract_code_requirements(code_samples)

    # generate candidates for each spec
    # code for candidates here ! ! !

if __name__ == "__main__":
    main()

