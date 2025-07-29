import os
from datasets import load_dataset


def download_dataset_from_hf(dataset_name: str, target_dir: str):
    """
    Downloads a dataset from Hugging Face to the specified directory.

    :param dataset_name: The name of the dataset on Hugging Face.
    :param target_dir: The directory where the dataset should be saved.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Save the dataset to the target directory
    dataset.save_to_disk(target_dir)

    print(f"Dataset '{dataset_name}' has been downloaded to '{target_dir}'.")


def convert_dataset_to_json(dataset, output_file):
    """
    Converts a dataset to JSON format and saves it to a file.

    :param dataset: The dataset to convert.
    :param output_file: The file path where the JSON should be saved.
    """
    # Convert the dataset to a pandas DataFrame
    df = dataset.to_pandas()
    
    # Save the DataFrame to a JSON file
    df.to_json(output_file, orient='records', lines=True)
    
    print(f"Dataset has been converted to JSON and saved to '{output_file}'.")


if __name__ == "__main__":
    # Example usage
    example_dataset_name = "BAAI/TACO"
    example_target_dir = "/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset"
    example_json_output = "/data/lishizheng/code/peft_study/datasets-peft/TACO/taco_dataset/taco_train.json"
    
    # Load the dataset from disks
    dataset = load_dataset("BAAI/TACO", split="test")
    
    # Convert the dataset to JSON
    convert_dataset_to_json(dataset, example_json_output)
