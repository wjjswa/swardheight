import os
import ast
import json


class Genotype:
    def __init__(self, normal, normal_concat, reduce, reduce_concat):
        self.normal = normal
        self.normal_concat = normal_concat
        self.reduce = reduce
        self.reduce_concat = reduce_concat

    def __repr__(self):
        return (f"Genotype(\n"
                f"    normal={self.normal},\n"
                f"    normal_concat={self.normal_concat},\n"
                f"    reduce={self.reduce},\n"
                f"    reduce_concat={self.reduce_concat}\n)"
                )
def load_genotype(file_path):
    """Load the genotype from a file."""
    with open(file_path, 'r') as f:
        genotype_dict = json.load(f)

    # Convert dict back to Genotype object
    return Genotype(
        normal=genotype_dict['normal'],
        normal_concat=genotype_dict['normal_concat'],
        reduce=genotype_dict['reduce'],
        reduce_concat=genotype_dict['reduce_concat']
    )
def list_files_in_directory(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist or is not a directory.")

    try:
        files = [os.path.join(directory, name) for name in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, name))]

        if not files:
            print(f"No files found in {directory}")

        return files
    except Exception as e:
        raise RuntimeError(f"An error occurred while listing files: {e}")

def load_data(file_dir):
    # test_data = torch.load(f"{file_dir}/test_data.pt")
    # test_targets = torch.load(f"{file_dir}/test_targets.pt")
    #
    # test_dataset = CustomDataset(test_data, test_targets)
    #
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    min_val = load_txt_file(f'{file_dir}/min.txt')
    max_val = load_txt_file(f'{file_dir}/max.txt')
    input_size = load_txt_file(f'{file_dir}/input_size.txt')
    fc_output = load_txt_file(f'{file_dir}/target_size.txt')
    in_channels = input_size[0]
    return  min_val, max_val, in_channels, fc_output



def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        shape_str = file.readline().strip()  # Read the first line and strip any extraneous whitespace/newlines
        shape_tuple = ast.literal_eval(shape_str)  # Safely evaluate the string as a tuple
    return shape_tuple