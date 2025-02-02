"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple, Dict
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse

def read_layout_data(data_dir: str) -> List[Dict]:
    """
    Read layout data from txt file and convert to list of dictionaries.
    Handles both orders:
    - [User prompt] ... [Layout] ...
    - [Layout] ... [User prompt] ...
    """
    data = []
    
    # Get all txt files in directory
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        raise ValueError(f"No txt files found in {data_dir}")
        
    print(f"Found {len(txt_files)} txt files")
    
    for txt_file in txt_files:
        file_path = os.path.join(data_dir, txt_file)
        print(f"Processing {txt_file}...")
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Extract prompt and layout regardless of order
                prompt_part = None
                layout_part = None
                
                # Find all parts using regex to handle both orders
                prompt_match = re.search(r'\[User prompt\](.*?)(?=\[Layout\]|\Z)', line)
                layout_match = re.search(r'\[Layout\](.*?)(?=\[User prompt\]|\Z)', line)
                
                if prompt_match and layout_match:
                    prompt_part = prompt_match.group(1).strip()
                    layout_part = layout_match.group(1).strip()
                    
                    # Add to data if we have both parts
                    if prompt_part and layout_part:
                        data.append({
                            'prompt': prompt_part,
                            'layout': layout_part
                        })
    
    print(f"Total examples loaded: {len(data)}")
    return data


def make_prefix(example, template_type='base'):
    """
    Create prompt prefix based on template type.
    Args:
        example: Dictionary containing 'prompt' key
        template_type: String indicating which template to use
    Returns:
        String: Formatted prompt
    """
    prompt = example['prompt']  # Extract prompt from example dictionary
    
    if template_type == 'base':
        return f"""A conversation between User and Assistant about architectural layout design. The Assistant is an expert architect who understands spatial relationships, room dimensions, and design principles. When given a constraint, the Assistant first analyzes the requirements, then plans the layout considering architectural best practices, and finally generates the geometric solution.

The layout should be specified using room labels followed by their coordinates. Each room can have 4 or more coordinates to define its shape (rectangular or more complex shapes like L or T). Coordinates are in a 256x256 grid system. For example:
- A rectangular bedroom: 'bedroom: (0,0)(100,0)(100,100)(0,100)'
- An L-shaped living room: 'living_room: (0,0)(200,0)(200,100)(100,100)(100,200)(0,200)'

Valid room labels are: bedroom, bathroom, living_room, kitchen, dining_room, corridor.
Rooms must connect through shared walls and form a coherent living space.

User: Create a layout that satisfies this constraint: {prompt}. First analyze the requirements in <requirements> </requirements> tags. Then plan the spatial relationships in <planning> </planning> tags. Finally, generate the layout coordinates in <layout> </layout> tags."""

    elif template_type == 'base-explicit':
        return f"""A conversation between User and Assistant about architectural layout design. The Assistant is an expert architect who understands spatial relationships, room dimensions, and design principles. The layout should be practical and livable, following basic architectural principles.

Key Layout Rules:
1. Use format 'room_label: (x1,y1)(x2,y2)(x3,y3)(x4,y4)...' for each room
2. Coordinates must be within 256x256 grid
3. Valid room labels: bedroom, bathroom, living_room, kitchen, dining_room, corridor
4. Rooms must share walls (perfect coordinate overlap)
5. Each room must have at least 4 coordinates
6. Rooms can be rectangular or have more complex shapes (L, T, etc.)

Example valid rooms:
bedroom: (100,100)(200,100)(200,200)(100,200)
living_room: (0,0)(200,0)(200,100)(100,100)(100,200)(0,200)
bathroom: (50,50)(100,50)(100,100)(50,100)

User: Create a layout that satisfies this constraint: {prompt}. 

First analyze the requirements in <requirements> </requirements> tags. Then plan the spatial relationships in <planning> </planning> tags. Finally, generate the layout coordinates in <layout> </layout> tags.""""""

    else:
        # Default template if none of the above match
        return f"""A conversation between User and Assistant about architectural layout design. The Assistant is an expert architect who understands spatial relationships, room dimensions, and design principles.

Layout format: 'room_label: (x1,y1)(x2,y2)(x3,y3)(x4,y4)...' within 256x256 grid.
Valid labels: bedroom, bathroom, living_room, kitchen, dining_room, corridor.

User: Create a layout that satisfies this constraint: {prompt}. 

First analyze the requirements in <requirements> </requirements> tags. Then plan the spatial relationships in <planning> </planning> tags. Finally, generate the layout coordinates in <layout> </layout> tags."""

def inspect_datasets(local_dir: str):
    """
    Load and inspect the first item from train and test datasets.
    
    Args:
        local_dir (str): Directory containing train.parquet and test.parquet
    """
    # Load datasets
    train_path = f"{local_dir}/train.parquet"
    test_path = f"{local_dir}/test.parquet"
    
    try:
        # Load using datasets library
        train_dataset = Dataset.from_parquet(train_path)
        test_dataset = Dataset.from_parquet(test_path)
        
        print("=== Training Dataset ===")
        print(f"Total samples: {len(train_dataset)}")
        print("\nFirst sample:")
        first_train = train_dataset[0]
        print("\nPrompt:")
        print(first_train['prompt'][0]['content'])  # First message in prompt list
        print("\nGround Truth Layout:")
        print(first_train['reward_model']['ground_truth']['layout'])
        
        print("\n=== Test Dataset ===")
        print(f"Total samples: {len(test_dataset)}")
        print("\nFirst sample:")
        first_test = test_dataset[0]
        print("\nPrompt:")
        print(first_test['prompt'][0]['content'])
        print("\nGround Truth Layout:")
        print(first_test['reward_model']['ground_truth']['layout'])
        
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")

def prepare_datasets(args):
    # Read data from txt file
    raw_data = read_layout_data(args.data_dir) 
    
    # Convert to Dataset
    raw_dataset = Dataset.from_list(raw_data)
    
    # Split into train and test
    assert len(raw_dataset) > args.train_size + args.test_size
    train_dataset = raw_dataset.select(range(args.train_size))
    test_dataset = raw_dataset.select(range(args.train_size, args.train_size + args.test_size))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "layout": example['layout'],
                "prompt": example['prompt']
            }
            data = {
                "data_source": "architectural_layouts",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "spatial_reasoning",
                "reward_model": {
                    "style": "architectural",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    # Map the processing function over datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Save to parquet files
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    
    # Save local files
    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    # If hdfs_dir is specified, copy individual files instead of the whole directory
    if args.hdfs_dir is not None:
        hdfs_train_path = os.path.join(args.hdfs_dir, 'train.parquet')
        hdfs_test_path = os.path.join(args.hdfs_dir, 'test.parquet')
        
        # Copy individual files
        copy(train_path, hdfs_train_path)
        copy(test_path, hdfs_test_path)
        
    inspect_datasets(local_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing txt files with layout data')
    parser.add_argument('--train_size', type=int, default=800, help='Size of training set')
    parser.add_argument('--test_size', type=int, default=200, help='Size of test set')
    parser.add_argument('--template_type', type=str, default='base-explicit', help='Type of prompt template')
    parser.add_argument('--local_dir', type=str, required=True, help='Local directory to save parquet files')
    parser.add_argument('--hdfs_dir', type=str, help='HDFS directory to copy files to')
    
    args = parser.parse_args()
    prepare_datasets(args)
