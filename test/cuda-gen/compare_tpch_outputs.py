import pandas as pd
import numpy as np
import os
import traceback # Import traceback for potentially better error reporting

def _is_col_int_like(series, tol=1e-8):
    """
    Checks if a numeric pandas Series contains only values that are
    effectively integers (within a small tolerance). Ignores NaNs.
    """
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return False # Not a numeric column
    series_valid = series.dropna()
    if series_valid.empty:
        return True # An empty or all-NaN column doesn't violate the integer condition
    # Check if all valid numbers are very close to their rounded value
    # Use np.isclose for floating point robustness (handles 2.999999999... == 3)
    # rtol=0 ensures we only use absolute tolerance 'atol' for this check
    are_close_to_int = np.isclose(series_valid, np.round(series_valid), rtol=0, atol=tol)
    return np.all(are_close_to_int)

def compare_numeric_columns(df1_numeric, df2_numeric, float_comparison_method, percent_diff_threshold, abs_tolerance_for_percent, float_tolerance):
    equal_nan = True # Treat NaNs as equal
    # --- CHOOSE COMPARISON METHOD ---
    if float_comparison_method == 'percent_diff':
        print(f"Info: Comparing numeric columns using percentage difference (Threshold={percent_diff_threshold*100}%, Abs fallback={abs_tolerance_for_percent})")
        # Calculate absolute difference
        diff = np.abs(df1_numeric - df2_numeric)
        # Calculate allowed difference based on percentage (using df2 as reference)
        # Use np.maximum with abs_tolerance to avoid issues with division by zero or near-zero
        # This effectively sets a minimum denominator for percentage calculation relevance
        denominator = np.maximum(np.abs(df2_numeric), abs_tolerance_for_percent)
        allowed_percent_diff = percent_diff_threshold * denominator

        # Check if difference is within allowed percentage OR within absolute fallback
        comparison_result_calc = (diff <= allowed_percent_diff) | (diff <= abs_tolerance_for_percent)

        # Handle NaNs based on the equal_nan flag
        if equal_nan:
            nan_mask = np.isnan(df1_numeric) & np.isnan(df2_numeric)
            comparison_result = comparison_result_calc | nan_mask
        else:
            comparison_result = comparison_result_calc
            # Ensure non-matching NaNs fail the comparison
            either_nan_mask = np.isnan(df1_numeric) | np.isnan(df2_numeric)
            # Define nan_mask if not already defined
            if 'nan_mask' not in locals():
                    nan_mask = np.zeros_like(df1_numeric, dtype=bool)
            # Where one is NaN but not both, comparison must be False
            comparison_result = np.where(either_nan_mask & ~nan_mask, False, comparison_result)

    elif float_comparison_method == 'isclose':
        print(f"Info: Comparing numeric columns using np.isclose (rtol={float_tolerance}, atol={float_tolerance})")
        comparison_result = np.isclose(df1_numeric, df2_numeric,
                                        rtol=float_tolerance, atol=float_tolerance, equal_nan=equal_nan)
    else:
        raise ValueError("Invalid float_comparison_method. Choose 'isclose' or 'percent_diff'.")
    
    return comparison_result

def compare_csv_files_fuzzy(file1_path, file2_path, float_tolerance=1e-5, percent_diff_threshold=0.005, 
                            abs_tolerance_for_percent=1e-5, sort_columns=None, test_has_header=False):
    """
    Compares two CSV files for equality, allowing for out-of-order rows
    and fuzzy comparison of floating-point numbers using np.isclose.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
        float_tolerance (float): Absolute and relative tolerance (`atol` and `rtol`)
                                for comparing floating-point numbers via np.isclose.
                                Defaults to 1e-5.
        sort_columns (list, optional): List of column names to sort by before comparison.
                                       If None, sorts by all columns. Defaults to None.

    Returns:
        bool: True if the CSV files are considered equal (considering row order
              and float tolerance), False otherwise.
        str: A message indicating the result or the reason for inequality.
    """
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1_path, header=0, delimiter='|')
        df2 = pd.read_csv(file2_path, header=0, delimiter='|')
        df2.columns = df1.columns # Set df2's columns to match df1's

        # --- Basic Checks ---
        # Check shape (number of rows and columns)
        if df1.shape != df2.shape:
            return False, f"DataFrames have different shapes: {df1.shape} vs {df2.shape}"

        # Check column names and order
        if not df1.columns.equals(df2.columns):
            # If columns are the same but just in a different order, reorder df2
            if sorted(df1.columns) == sorted(df2.columns):
                print(f"Info: Columns in '{os.path.basename(file2_path)}' reordered to match '{os.path.basename(file1_path)}' for comparison.")
                df2 = df2[df1.columns] # Reorder columns in df2 to match df1
            else:
                # Identify missing/extra columns
                cols1 = set(df1.columns); cols2 = set(df2.columns)
                diff = (cols1 - cols2) | (cols2 - cols1)
                return False, f"DataFrames have different columns. Difference: {diff}"

        # --- Handle Out-of-Order Rows by Sorting ---
        if sort_columns:
            # Ensure provided sort columns exist in the DataFrame
            if not all(col in df1.columns for col in sort_columns):
                 missing_cols = [col for col in sort_columns if col not in df1.columns]
                 return False, f"Specified sort columns not found in DataFrames: {missing_cols}"
            sort_by = sort_columns
        else:
            # Default to sorting by all columns if none specified
            sort_by = list(df1.columns)

        print(f"Info: Sorting by columns: {sort_by}")

        # Convert potentially mixed-type object columns to string before sorting
        # This avoids errors when trying to sort columns with e.g. numbers and text
        for col in sort_by:
             if df1[col].dtype == 'object':
                 try:
                     # Convert only if not all null to avoid converting columns of NaNs
                     if not df1[col].isnull().all(): df1[col] = df1[col].astype(str)
                 except (TypeError, ValueError, OverflowError): pass # Ignore conversion errors
             if df2[col].dtype == 'object':
                 try:
                     if not df2[col].isnull().all(): df2[col] = df2[col].astype(str)
                 except (TypeError, ValueError, OverflowError): pass

        try:
            # Perform sorting and reset index to align rows for comparison
            df1_sorted = df1.sort_values(by=sort_by).reset_index(drop=True)
            df2_sorted = df2.sort_values(by=sort_by).reset_index(drop=True)
        except Exception as e:
             # Catch potential type errors during sorting (e.g., comparing str and float)
             return False, f"Error during sorting: {e}. Ensure columns used for sorting ({sort_by}) have comparable types."


        # --- Compare Data ---
        # Separate numeric and non-numeric columns
        numeric_cols_df = df1_sorted.select_dtypes(include=np.number)
        numeric_cols = numeric_cols_df.columns
        other_cols = df1_sorted.select_dtypes(exclude=np.number).columns

        # 1. Compare non-numeric columns exactly (handles NaNs correctly)
        if not other_cols.empty:
            # Overall check first for efficiency
            if not df1_sorted[other_cols].equals(df2_sorted[other_cols]):
                # If they differ, find the *first* differing column and row
                first_diff_col_name = None
                first_diff_row_idx = -1
                val1, val2 = None, None

                # Iterate through non-numeric columns to find the first one with differences
                for col in other_cols:
                    col1_series = df1_sorted[col]
                    col2_series = df2_sorted[col]
                    if not col1_series.equals(col2_series): # Check if the Series are different
                        first_diff_col_name = col
                        # Find first index where values are different, respecting NaNs
                        # (NaN != NaN is True, so we exclude cases where both are NaN)
                        diff_mask = (col1_series != col2_series) & ~(col1_series.isna() & col2_series.isna())
                        if diff_mask.any(): # Check if there's any True difference after NaN handling
                             first_diff_row_idx = diff_mask.idxmax() # Get index of first True
                             val1 = df1_sorted.loc[first_diff_row_idx, first_diff_col_name]
                             val2 = df2_sorted.loc[first_diff_row_idx, first_diff_col_name]
                        else:
                             # If .equals was False but diff_mask is all False,
                             # the difference must be NaN vs non-NaN
                             isnull_diff_mask = col1_series.isna() != col2_series.isna()
                             if isnull_diff_mask.any():
                                 first_diff_row_idx = isnull_diff_mask.idxmax()
                                 val1 = df1_sorted.loc[first_diff_row_idx, first_diff_col_name]
                                 val2 = df2_sorted.loc[first_diff_row_idx, first_diff_col_name]
                             else: # Should not happen, fallback
                                 first_diff_row_idx = -99 # Indicate failure to pinpoint
                                 val1 = "N/A"
                                 val2 = "N/A"
                        break # Stop after finding the first differing column

                if first_diff_col_name is not None and first_diff_row_idx != -99:
                    return False, (f"Non-numeric data differs. First difference found at "
                                   f"sorted row index {first_diff_row_idx}, column '{first_diff_col_name}': "
                                   f"'{val1}' (file 1) vs '{val2}' (file 2)")
                else:
                    # Fallback message if exact location wasn't determined
                    return False, "Non-numeric data differs, but failed to locate the exact first difference."

        integer_like_cols = []
        float_like_cols = []
        if not numeric_cols.empty:
            int_check_tolerance = 1e-8 # Tolerance for integer-like check
            print("Info: Classifying numeric columns as 'integer-like' (exact compare) or 'float-like' (fuzzy compare)...")
            for col in numeric_cols:
                # Must be integer-like in BOTH dataframes to qualify for exact comparison
                is_int_like1 = _is_col_int_like(df1_sorted[col], tol=int_check_tolerance)
                is_int_like2 = _is_col_int_like(df2_sorted[col], tol=int_check_tolerance)
                if is_int_like1 and is_int_like2:
                    integer_like_cols.append(col)
                else:
                    float_like_cols.append(col)
            print(f"Info: Integer-like numeric columns (exact compare): {integer_like_cols}")
            print(f"Info: Float-like numeric columns (fuzzy compare): {float_like_cols}")

        # 2a. Compare Integer-Like Numeric Columns Exactly
        if len(integer_like_cols) > 0:
            df1_int_like = df1_sorted[integer_like_cols]
            df1_int_like = df1_int_like.astype('Int64') # Use 'Int64' for nullable integers
            df2_int_like = df2_sorted[integer_like_cols]
            df2_int_like = df2_int_like.astype('Int64') # Use 'Int64' for nullable integers
            if not df1_int_like.equals(df2_int_like):
                print("Info: Difference found in integer-like numeric columns.")
                # Find the first difference (using similar logic as non-numeric)
                first_diff_col_name = None
                first_diff_row_idx = -1
                val1, val2 = None, None
                for col in integer_like_cols: # Iterate through only these columns
                    if not df1_int_like[col].equals(df2_int_like[col]):
                        first_diff_col_name = col
                        diff_mask = (df1_int_like[col] != df2_int_like[col]) & \
                                    ~(df1_int_like[col].isna() & df2_int_like[col].isna())
                        if diff_mask.any():
                            first_diff_row_idx = diff_mask.idxmax()
                            val1 = df1_int_like.loc[first_diff_row_idx, first_diff_col_name]
                            val2 = df2_int_like.loc[first_diff_row_idx, first_diff_col_name]
                        else: # NaN vs non-NaN difference
                            isnull_diff_mask = df1_int_like[col].isna() != df2_int_like[col].isna()
                            if isnull_diff_mask.any():
                                first_diff_row_idx = isnull_diff_mask.idxmax()
                                val1 = df1_int_like.loc[first_diff_row_idx, first_diff_col_name]
                                val2 = df2_int_like.loc[first_diff_row_idx, first_diff_col_name]
                            else: first_diff_row_idx = -99 # Should not happen
                        break # Found first differing column

                if first_diff_col_name is not None and first_diff_row_idx != -99:
                     # Print the differing rows from both dataframes for better debugging
                     print(f"Differing rows at index {first_diff_row_idx}:")
                     print(f"DF1: {df1_sorted.iloc[first_diff_row_idx].to_dict()}")
                     print(f"DF2: {df2_sorted.iloc[first_diff_row_idx].to_dict()}")
                     return False, (f"Integer-like numeric data differs. First difference found at "
                                   f"sorted row index {first_diff_row_idx}, column '{first_diff_col_name}': "
                                   f"{val1} (file 1) vs {val2} (file 2)")
                else:
                     return False, "Integer-like numeric data differs, but failed to locate the exact first difference."

        if len(float_like_cols) > 0:
            try:
                # Convert to numpy arrays for efficient comparison
                df1_numeric = df1_sorted[float_like_cols].to_numpy()
                df2_numeric = df2_sorted[float_like_cols].to_numpy()

                # Perform element-wise comparison with tolerance, treating NaNs as equal
                comparison_result = compare_numeric_columns(df1_numeric, df2_numeric, 'percent_diff', 
                                                            percent_diff_threshold=percent_diff_threshold, 
                                                            abs_tolerance_for_percent=float_tolerance,
                                                            float_tolerance=float_tolerance,)

                # Check if all elements were close enough
                if not np.all(comparison_result):
                    # If not, find the location [row, column] of the first difference
                    diff_indices = np.argwhere(~comparison_result)
                    first_diff_row_idx, first_diff_col_idx = diff_indices[0]
                    # Map numpy column index back to DataFrame column name
                    first_diff_col_name = float_like_cols[first_diff_col_idx]
                    # Get the actual values from the sorted DataFrames
                    val1 = df1_sorted.loc[first_diff_row_idx, first_diff_col_name]
                    val2 = df2_sorted.loc[first_diff_row_idx, first_diff_col_name]
                    return False, (f"Numeric data differs beyond tolerance (atol={float_tolerance}, rtol={float_tolerance}). "
                                   f"First difference found at sorted row index {first_diff_row_idx}, "
                                   f"column '{first_diff_col_name}': {val1} (file 1) vs {val2} (file 2)")
            except Exception as e:
                 # Catch potential errors during numpy conversion or comparison
                 return False, f"Error during numeric comparison: {e}"

        # If all non-numeric and numeric checks passed
        return True, "CSV files are considered equal (handling row order and float tolerance)."

    except FileNotFoundError as e:
        return False, f"Error: File not found - {e}"
    except pd.errors.EmptyDataError:
         # Refined empty file check
         df1_check, df2_check = None, None
         df1_empty, df2_empty = True, True
         try:
             # Attempt to read, check if empty
             df1_check = pd.read_csv(file1_path)
             if not df1_check.empty: df1_empty = False
         except pd.errors.EmptyDataError: pass # File is empty or header-only
         except FileNotFoundError: pass # Already caught above, but handle race conditions

         try:
             df2_check = pd.read_csv(file2_path)
             if not df2_check.empty: df2_empty = False
         except pd.errors.EmptyDataError: pass
         except FileNotFoundError: pass

         # Determine outcome based on emptiness
         if df1_empty and df2_empty:
             return True, "Both CSV files are empty or could not be read (potentially header-only or missing)."
         elif df1_empty or df2_empty:
             return False, "One CSV file is empty/unreadable, the other is not."
         else:
             # This case might occur if read_csv succeeded initially but EmptyDataError was raised later? Unlikely.
             return False, "Error handling potentially empty files."

    except Exception as e:
        # Catch-all for any other unexpected errors
        print("--- Unexpected Error Traceback ---")
        print(traceback.format_exc()) # Print detailed traceback for debugging
        print("---------------------------------")
        return False, f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    import sys
    
    if (len(sys.argv) < 3):
        print("Usage: python compare_csv.py <csv_file1> <csv_file2> [--test-has-header]")
        sys.exit(1)
          
    test_has_header = False
    if len(sys.argv) == 4 and sys.argv[3] == '--test-has-header':
        test_has_header = True
    
    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]
    float_tolerance = 1e-5 # Default tolerance for float comparison
    are_equal, message = compare_csv_files_fuzzy(csv_file1, csv_file2, float_tolerance=float_tolerance, test_has_header=test_has_header) 
    print(f"Comparison result: {are_equal}")
    print(f"Message: {message}")
    
    # if are_equal is False, exit with non-zero status
    if not are_equal:
        sys.exit(1)
     
    def test_compare():
        # --- Example Usage (as used in the last execution) ---
        # Create dummy CSV files for demonstration
        data1 = {'ID': [1, 2, 3, 4],
                'Name': ['Alice', 'Bob', 'Charlie', 'David'],
                'Value': [100.0, 200.5, 300.99999, 400.12345],
                'Category': ['A', 'B', 'A', 'C']}
        df_a = pd.DataFrame(data1)

        # File 2: Different row order, floats fuzzily equal, different non-float ('Category')
        data2 = {'ID': [3, 1, 4, 2],
                'Name': ['Charlie', 'Alice', 'David', 'Bob'],
                'Value': [300.999991, 100.000001, 400.123456, 200.500004],
                'Category': ['X', 'A', 'C', 'B']} # 'X' differs from 'A' for Charlie
        df_b = pd.DataFrame(data2)

        # File 3: Different row order, floats fuzzily equal, non-floats equal
        data3 = {'ID': [4, 1, 3, 2],
                'Name': ['David', 'Alice', 'Charlie', 'Bob'],
                'Value': [400.1234501, 100.0000009, 300.999999, 200.5000005],
                'Category': ['C', 'A', 'A', 'B']}
        df_c = pd.DataFrame(data3)

        # File 4: Same as File 1 but one float is significantly different (Value: 301.1)
        data4 = {'ID': [1, 2, 3, 4],
                'Name': ['Alice', 'Bob', 'Charlie', 'David'],
                'Value': [100.0, 200.5, 301.1, 400.12345],
                'Category': ['A', 'B', 'A', 'C']}
        df_d = pd.DataFrame(data4)

        # Save to CSV
        csv_path1 = 'file_A.csv'
        csv_path2 = 'file_B_diff_cat.csv'
        csv_path3 = 'file_C_equal.csv'
        csv_path4 = 'file_D_diff_val.csv'
        df_a.to_csv(csv_path1, index=False)
        df_b.to_csv(csv_path2, index=False)
        df_c.to_csv(csv_path3, index=False)
        df_d.to_csv(csv_path4, index=False)

        print(f"Created dummy CSV files: {csv_path1}, {csv_path2}, {csv_path3}, {csv_path4}\n")

        # --- Comparison Examples ---

        # 1. Compare A and C (should be equal with default tolerance 1e-5)
        print(f"--- Comparing '{csv_path1}' and '{csv_path3}' (tolerance=1e-5) ---")
        are_equal, message = compare_csv_files_fuzzy(csv_path1, csv_path3, float_tolerance=1e-5)
        print(f"Result: {are_equal}")
        print(f"Message: {message}\n")

        # 2. Compare A and B (should be different due to 'Category')
        print(f"--- Comparing '{csv_path1}' and '{csv_path2}' (tolerance=1e-5) ---")
        are_equal, message = compare_csv_files_fuzzy(csv_path1, csv_path2, float_tolerance=1e-5)
        print(f"Result: {are_equal}")
        print(f"Message: {message}\n")

        # 3. Compare A and D (should be different due to 'Value' change)
        print(f"--- Comparing '{csv_path1}' and '{csv_path4}' (tolerance=1e-5) ---")
        are_equal_strict, message_strict = compare_csv_files_fuzzy(csv_path1, csv_path4, float_tolerance=1e-5)
        print(f"Result: {are_equal_strict}")
        print(f"Message: {message_strict}\n")

        # 4. Compare A and C with very strict tolerance (should FAIL now on floats)
        print(f"--- Comparing '{csv_path1}' and '{csv_path3}' with stricter tolerance (1e-8) ---")
        are_equal_strict_2, message_strict_2 = compare_csv_files_fuzzy(csv_path1, csv_path3, float_tolerance=1e-8)
        print(f"Result: {are_equal_strict_2}")
        print(f"Message: {message_strict_2}\n")

        # 5. Compare using specific columns for sorting (e.g., 'ID')
        print(f"--- Comparing '{csv_path1}' and '{csv_path3}' sorting only by 'ID' ---")
        are_equal_sort_id, message_sort_id = compare_csv_files_fuzzy(csv_path1, csv_path3, sort_columns=['ID'])
        print(f"Result: {are_equal_sort_id}")
        print(f"Message: {message_sort_id}\n")

        # Clean up dummy files
        try:
            os.remove(csv_path1)
            os.remove(csv_path2)
            os.remove(csv_path3)
            os.remove(csv_path4)
            print("Cleaned up dummy CSV files.")
        except OSError as e:
            print(f"Error removing dummy files: {e}")