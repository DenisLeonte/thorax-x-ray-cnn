import os
import pandas as pd
from datetime import datetime

def log_experiment(output_file, model_name, hyperparameters, metrics):
    """
    Logs experiment results to an Excel file.
    
    Args:
        output_file (str): Path to the Excel file.
        model_name (str): Name of the model architecture.
        hyperparameters (dict): Dictionary of hyperparameters (lr, batch_size, etc.).
        metrics (dict): Dictionary of metrics (loss, accuracy, auc, etc.).
    """
    
    # Prepare the data row
    data = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': model_name,
        **hyperparameters,
        **metrics
    }
    
    df_new = pd.DataFrame([data])
    
    if os.path.exists(output_file):
        try:
            # Try loading existing excel
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # Load existing sheet to find the last row
                try:
                    df_existing = pd.read_excel(output_file)
                    start_row = len(df_existing) + 1
                    header = False
                except ValueError: 
                    # Sheet might not exist or file is empty
                    start_row = 0
                    header = True
                    
                df_new.to_excel(writer, index=False, header=header, startrow=start_row)
        except Exception as e:
            print(f"Error appending to Excel: {e}. Creating a backup or new file.")
            # Fallback: just append to CSV if Excel fails logic, but requested was Excel. 
            # We will try to re-write the whole file if append fails (simpler for pandas sometimes)
            df_existing = pd.read_excel(output_file)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(output_file, index=False)
            
    else:
        # Create new file
        df_new.to_excel(output_file, index=False)
    
    print(f"Logged experiment results to {output_file}")
