import yaml
import argparse
import re
import os

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_filters(filter_str):
    """Parse filter string into a list of filter conditions"""
    if not filter_str:
        return []
    
    # Remove parentheses if present
    filter_str = filter_str.strip('()')
    
    # Split into individual conditions
    conditions = [cond.strip() for cond in filter_str.split(',')]
    
    filters = []
    for cond in conditions:
        # Match comparison operators
        match = re.match(r'(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)', cond)
        if match:
            key, op, value = match.groups()
            filters.append(('compare', key.strip(), op, value.strip()))
        else:
            # Default to equality check if no operator specified
            if '=' in cond:
                key, value = cond.split('=', 1)
                filters.append(('equal', key.strip(), value.strip()))
    
    return filters

def apply_filters(videos, datasets, filters):
    filtered = []
    
    for video in videos:
        # Check dataset name
        if datasets:
            matches_dataset = False
            for dataset in datasets:
                if video['video'].startswith(dataset + '_'):
                    matches_dataset = True
                    break
            if not matches_dataset:
                continue
        
        # Apply all filters
        match = True
        for filter_type, *args in filters:
            if filter_type == 'equal':
                key, value = args
                if key not in video['tags'] or str(video['tags'][key]) != value:
                    match = False
                    break
            elif filter_type == 'compare':
                key, op, value = args
                if key not in video['tags']:
                    match = False
                    break
                
                try:
                    tag_value = float(video['tags'][key]) if '.' in video['tags'][key] else int(video['tags'][key])
                    filter_value = float(value) if '.' in value else int(value)
                except ValueError:
                    # If not numeric, compare as strings
                    tag_value = str(video['tags'][key])
                    filter_value = str(value)
                
                if op == '==' and tag_value != filter_value:
                    match = False
                    break
                elif op == '!=' and tag_value == filter_value:
                    match = False
                    break
                elif op == '<' and tag_value >= filter_value:
                    match = False
                    break
                elif op == '>' and tag_value <= filter_value:
                    match = False
                    break
                elif op == '<=' and tag_value > filter_value:
                    match = False
                    break
                elif op == '>=' and tag_value < filter_value:
                    match = False
                    break
        
        if match:
            filtered.append(video['video'])
    
    return filtered

def save_list(file_path, video_list):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(video_list))

def main():
    parser = argparse.ArgumentParser(description='Filter videos from YAML metadata')
    parser.add_argument('list_name', help='Name of the list to create')
    parser.add_argument('--datasets', nargs='+', default=None, help='Dataset names to filter by (space-separated, optional)')
    parser.add_argument('--filter', help='Filter conditions (e.g., "(classe=A3, duration>10)"')
    args = parser.parse_args()

    output_file = "../../data/lists/" + args.list_name + ".txt"

    # Load YAML file (assuming it's in the same location as before)
    yaml_file = '../../data/datasets/meta_video.yml'
    videos = load_yaml(yaml_file)
    
    # Parse filters
    filters = parse_filters(args.filter) if args.filter else []
    
    # Apply filters
    filtered_videos = apply_filters(videos, args.datasets, filters)
    
    # Save results
    save_list(output_file, filtered_videos)
    
    print(f"Saved {len(filtered_videos)} videos to {output_file}")

if __name__ == '__main__':
    main()