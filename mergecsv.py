# import pandas as pd

# # File paths for your CSV files
# file_paths = ['datas/1.csv', 'datas/3.csv', 'datas/4.csv']

# # Read each CSV file into a DataFrame
# data_frames = [pd.read_csv(file) for file in file_paths]

# # Concatenate the DataFrames
# combined_data = pd.concat(data_frames)

# # Write the combined data to a new CSV file
# combined_data.to_csv('134.csv', index=False)

# print("Files appended and combined successfully into 'combined_file.csv'")

import pandas as pd

# Load the CSV file into a DataFrame
file_path = '134.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# List of keywords to filter
lst = ['semen', 'masturbating', 'masturbate','ejaculate', 'FORESKINS','mastrubate','tight under wear']  # Replace with your keywords

# Filter rows based on keywords in Query or Answer
filtered_data = data[~(data['Query'].str.contains('|'.join(lst), case=False) | data['Answer'].str.contains('|'.join(lst), case=False))]

# Save the filtered data to a new CSV file
filtered_data.to_csv('134_filtered.csv', index=False)

print("Pairs containing specified keywords removed and saved to 'filtered_file.csv'")

