'''
Convert two CSV files in the structure
Song ID, A_AVG, A_SD and Song ID, V_AVG, V_SD

to a new combined file that is in the structure
Song ID, A_AVG, A_SD, V_AVG, V_SD
'''
import pandas as pd

# Open CSV files
a_file = "data/annotations_new/arousal_cont_10.csv"
v_file = "data/annotations_new/valence_cont_10.csv"

a_columns = ["SONG_ID", "AROUSAL_AVG", "AROUSAL_SD"]
v_columns = ["SONG_ID", "VALENCE_AVG", "VALENCE_SD"]

# Load the CSV files into pandas dataframes
a_df = pd.read_csv(a_file, names=a_columns, header=None)
v_df = pd.read_csv(v_file, names=v_columns, header=None)

print(a_df.shape)
print(v_df.shape)

# Merge the dataframes on the 'Song ID' column
merged_df = pd.merge(a_df, v_df, on='SONG_ID')

# Save the merged dataframe to a new CSV file
output_file = "data/annotations_new/combined_10.csv"
merged_df.to_csv(output_file, index=False)

print(merged_df.shape)

print(f"Merged data saved to {output_file}")


'''
FORMAT PREVIEW
|---------|---------------------|--------------------|-------------------------|--------------------|
| SONG_ID | AROUSAL_AVG         | AROUSAL_SD         | VALENCE_AVG             | VALENCE_SD         |
|---------|---------------------|--------------------|-------------------------|--------------------|
| 2_1     | -0.1382611380190284 | 0.0137419137914778 |     -0.0867074593435631 | 0.0099225905955622 |
|---------|---------------------|--------------------|-------------------------|--------------------|
| 2_2     |  -0.195654348247604 | 0.0539337774692676 |     -0.2447585835521885 | 0.0461736829797123 |
|---------|---------------------|--------------------|-------------------------|--------------------|
| 2_3     | -0.2629175792093376 | 0.0038846410867395 |     -0.3205412628014729 | 0.0099923409490446 |
|---------|---------------------|--------------------|-------------------------|--------------------|
| 3_1     | -0.1543442569856485 | 0.0165118243164503 |     -0.2100827905802743 | 0.0195292038078321 |
|---------|---------------------|--------------------|-------------------------|--------------------|
'''