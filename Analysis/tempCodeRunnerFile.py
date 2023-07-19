 row in file1_df.iloc[1:].iterrows(): #skip the first row because it is the header
        #     if row.notna().any(): # check if at least one cell is filled
        #         for header, value in zip(file1_df.columns, row):
        #             if "Right" in header:
        #                 if value != None: #something is detected for the right hand
        #                     if header in all_keys: #this is something we want to transfer to the new csv
        #                         d_frame[header] = value
        #                     # print(header, value)
        #     d_frame["Index"] = index
        # print(d_frame)