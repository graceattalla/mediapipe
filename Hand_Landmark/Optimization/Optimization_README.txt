All optimization files are currently based on the total number of rows in the csv file which also counts "None" rows. 

For all tested parameters the skeleton still showed that there was inadequate amounts of data.

Use this: non_none_rows = output_df.notna().any(axis=1).sum()