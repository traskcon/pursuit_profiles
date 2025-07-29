import os
import numpy as np
import pandas as pd
import pickle

from data_cleaning import clean_data, optimize_memory_usage
from data_preprocessing import aggregate_data, annotate_missed_tackle_frames, annotate_tackle_frames

# 1. Load and prepare data -----------------------------------------------------

root_dir = os.getcwd()
load_saved_data = True

if not load_saved_data:
    plays_fname = os.path.join(root_dir, "data/plays.csv")
    players_fname = os.path.join(root_dir, "data/players.csv")
    tackles_fname = os.path.join(root_dir, "data/tackles.csv")
    tracking_fname_list = [os.path.join(root_dir, f"data/tracking_week_1.csv")]

    # Aggregate data from the plays.csv, players.csv, tackles.csv, and any tracking data into one aggregate dataframe.
    df = aggregate_data(plays_fname, tackles_fname, players_fname, tracking_fname_list)

    # Preprocess and clean the data
    df_clean = clean_data(df)

    # optimize memory usage of dataframe
    df_opt = optimize_memory_usage(df_clean)

    # Annotate the frameId of both made tackles and missed tackles.
    # Note that currently, out-of-bounds tackles are not included, nor are assisted tackles (2 assists, 0 tackles)
    #df_tackles = pd.read_csv(tackles_fname)
    #df_tackles, tackle_ct = annotate_tackle_frames(df_opt, df_tackles)
    #df_tackles, missed_tackle_ct = annotate_missed_tackle_frames(df_opt, df_tackles)

    with open("./data/test_tracking_data.pkl", "wb") as f: 
        pickle.dump(df_opt, f)
else:
    df_opt = pickle.load(open(os.path.join(root_dir,"data/test_tracking_data.pkl"),"rb"))

# 2. Visualize single play -----------------------------------------------------
def calculate_pursuit_metrics(df_frame: pd.DataFrame, tacklerId: int) -> tuple:
    # For a single frame of tracking data, calculate a given tackler's pursuit metrics
    ballCarrierId = df_frame.ballCarrierId.iloc[0]
    # Pursuit speed (yards/s)
    s_p = df_frame[df_frame.nflId == tacklerId].s_clean.values[0]/100
    # Pursuit angle (deg)
    pos_bc = df_frame[df_frame.nflId == ballCarrierId][["x_clean","y_clean"]].values[0]/100
    pos_p = df_frame[df_frame.nflId == tacklerId][["x_clean","y_clean"]].values[0]/100
    theta_rel = np.arctan2(pos_bc[1] - pos_p[1], pos_bc[0] - pos_p[0])*180/np.pi
    theta_rel = (theta_rel + 360) % 360
    p_heading = df_frame[df_frame.nflId == tacklerId]["dir_clean"].values[0]/10
    theta_p = np.abs(p_heading - theta_rel)
    return s_p, theta_p

game_id = 2022091200
play_id = 3826
tackler_id = 42827

df_play = df_opt[(df_opt["gameId"] == game_id)&(df_opt["playId"] == play_id)]
for frame in pd.unique(df_play.frameId.values):
    df_frame = df_play[df_play["frameId"] == frame]
    print("Frame: {}, v_p: {}, Angle: {:.4f}".format(frame,
                            *calculate_pursuit_metrics(df_frame, tackler_id)))