save_path = inner_folder + "\\" + f"B&B_{int(percent_filled)}%_" + file_name
    
    bb_mediapipe_df.to_csv(save_path)