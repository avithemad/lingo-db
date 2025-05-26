import pandas as pd
import io

def parse_single_result_file(file_path):
    """
    Parses a single result file and returns a DataFrame with the parsed data.
    
    Args:
        file_path (str): Path to the result file.
        
    Returns:
        pd.DataFrame: DataFrame containing the parsed data.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    
    data_io = io.StringIO(data)

    entries = []
    current_entry_lines = []

    for line in data_io:
        line = line.strip()
        if line == "---":
            if current_entry_lines: # Process the previous entry
                query_name = current_entry_lines[0]
                total_time = 0.0
                sum_main_kernels = 0.0
                sum_count_kernels = 0.0

                for data_line in current_entry_lines[1:]:
                    parts = data_line.split(',')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            if key.startswith("main_"):
                                sum_main_kernels += value
                            elif key.startswith("count_"): # Assuming 'count' kernels would follow this naming
                                sum_count_kernels += value
                            elif key == "total_query":
                                total_time = value
                        except ValueError:
                            print(f"Warning: Could not parse value for '{key}' in query '{query_name}'")
                            continue
                
                total_kernel_time = sum_main_kernels + sum_count_kernels
                overheads = total_time - total_kernel_time
                
                entries.append({
                    "Query name": query_name,
                    "Total time": total_time,
                    "Sum of main kernels": sum_main_kernels,
                    "Sum of count kernels": sum_count_kernels,
                    "Total kernel execution time": total_kernel_time,
                    "Overheads": overheads,
                    "Overheads (%)": (overheads / total_time) * 100.0
                })
            current_entry_lines = [] # Start a new entry
        elif line: # Add non-empty lines to the current entry
            current_entry_lines.append(line)

    # Process the last entry after the loop finishes
    if current_entry_lines:
        query_name = current_entry_lines[0]
        total_time = 0.0
        sum_main_kernels = 0.0
        sum_count_kernels = 0.0

        for data_line in current_entry_lines[1:]:
            parts = data_line.split(',')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    if key.startswith("main_"):
                        sum_main_kernels += value
                    elif key.startswith("count_"):
                        sum_count_kernels += value
                    elif key == "total_query":
                        total_time = value
                except ValueError:
                    print(f"Warning: Could not parse value for '{key}' in query '{query_name}'")
                    continue
                    
        total_kernel_time = sum_main_kernels + sum_count_kernels
        overheads = total_time - total_kernel_time
        print(f"Processing query: {query_name}")
        print(type(overheads), overheads)
        print(type(total_time), total_time)

        
        entries.append({
            "Query name": query_name,
            "Total time": total_time,
            "Sum of main kernels": sum_main_kernels,
            "Sum of count kernels": sum_count_kernels,
            "Total kernel execution time": total_kernel_time,
            "Overheads": overheads,
            "Overheads (%)": (overheads / total_time) * 100.0
        })

    df = pd.DataFrame(entries)

    print("Generated DataFrame:")
    print(df)
    return df

def parse_all_result_files(dir_name):
    """
    Parses all result files in the specified directory and returns a DataFrame with the parsed data.
    
    Args:
        dir_name (str): Directory containing the result files.
        
    Returns:
        pd.DataFrame: DataFrame containing the parsed data from all files.
    """
    import os
    all_entries = []

    for filename in os.listdir(dir_name):
        # if the filename starts with 'tpch' or 'ssb' and ends with '.csv'
        if not filename.startswith(('tpch', 'ssb')):
            continue
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_name, filename)
            print(f"Parsing file: {file_path}")
            df = parse_single_result_file(file_path)
            all_entries.append(df)

    combined_df = pd.concat(all_entries, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    # get the directory name of the current script
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # parse all result files in the current directory
    df = parse_all_result_files(current_dir)
    
    # Save the DataFrame to a CSV file
    output_csv_file = "parsed_query_data.csv"
    df.to_csv(output_csv_file, index=False)
    # print(f"\nDataFrame saved to {output_csv_file}")