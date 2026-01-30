import yaml
import time

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.load_all(file, Loader=yaml.FullLoader)
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

def save_yaml_file(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

# Example usage
if __name__ == '__main__':
    # Load YAML data from a file
    #yaml.load(b'!!python/object/new:os.system [echo EXPLOIT!]',Loader=yaml.FullLoader)
    yaml_data = load_yaml_file('data.yaml')
    print("Loaded YAML data:")
    print(yaml_data)
    print()

    # Modify the data
    yaml_data['new_key'] = 'new_value'

    # Save the modified data to a new file
    save_yaml_file(yaml_data, 'modified_data.yaml')
    print("Modified YAML data saved to modified_data.yaml")

