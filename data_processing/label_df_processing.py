import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns 

import os, glob, shutil
from datetime import datetime

def process_df(df, user_id):
    # fillna  filename
    df = df.fillna(method='ffill')
    
    if user_id == "user_id_63513":
        index = [0,32,64]
        df.drop(index, inplace=True)
        
    # remove label have AM,PM in time string
    index = df.loc[df['Start Time'].str.contains("AM") | df['Start Time'].str.contains("PM") | df['End Time'].str.contains("AM") | df['End Time'].str.contains("PM")].index
    df.drop(index, inplace=True)
    
    # convert time columns
    df['start_time'] = df['Start Time'].apply(lambda x: datetime.strptime(x.strip(), '%H:%M:%S'))
    df['end_time'] = df['End Time'].apply(lambda x: datetime.strptime(x.strip(), '%H:%M:%S'))
    df['Duration'] = df['end_time'] - df['start_time']
    df['Duration'] = df['Duration'].apply(lambda x: x.total_seconds())
    df['start_time'] = df['start_time'].apply(lambda x: x.time())
    df['end_time'] = df['end_time'].apply(lambda x: x.time())
    df.sort_values(by=["Filename","end_time"], ascending=True, inplace=True)
    
    # filtering
    len_before = df.shape[0]
    typo_df = df.copy()
    typo_df = typo_df.loc[(typo_df['Duration'] <= 0) | (typo_df['Label (Primary)'] == 'Class ')]
    
    df = df.loc[(df['Duration'] > 0) & (df['Label (Primary)'] != 'Class ')].reset_index(drop=True)
    df['Class'] = df['Label (Primary)'].apply(lambda x: int(x.split()[-1]))

    len_after = df.shape[0]
    # log typo cases
    if(len_before != len_after):
        print("Typo cases")
        print(typo_df)

    # log number of actions in each view
    print("Number of actions in each view:")
    print(df.groupby("Camera View")["Filename"].count())
    
    return df


if __name__ == "__main__":
    path_setA1 = "../data/SetA1"
    user_foldersA1 = os.listdir(path_setA1)
    print("Number of users: ", len(os.listdir(path_setA1)))

    processed_dict = dict()
    output = "../data/SetA1/processed_label_csv"

    if os.path.exists(output) is False:
        os.makedirs(output)
        
    for id in range(len(user_foldersA1)):
        # read df
        user_id = user_foldersA1[id]
        print("="*20)
        print("User ", user_id)
        try:
            df = pd.read_csv(f"{path_setA1}/{user_id}/{user_id}.csv")
            df = process_df(df, user_id)

        except Exception as e:
            print(f"{user_id} - Error: {e}")
            continue
        
        df.to_csv(f"{output}/{user_id}.csv", index=False)
        processed_dict[user_id] = df
