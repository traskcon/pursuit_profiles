import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import plotly.express as px

from data_cleaning import clean_data, optimize_memory_usage
from data_preprocessing import aggregate_data, restore_geometry

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

# 2. Calculate Metrics for a single player, single frame -----------------------
def calculate_pursuit_metrics(df_frame: pd.DataFrame, tacklerId: int) -> tuple:
    # DEPRECIATED, DELETE LATER
    # For a single frame of tracking data, calculate a given tackler's pursuit metrics
    ballCarrierId = df_frame.ballCarrierId.iloc[0]
    tackle_data = df_frame[df_frame.nflId == tacklerId]
    # Pursuit speed (yards/s)
    s_p = tackle_data.s_clean.values[0]
    # Pursuit angle (deg)
    pos_bc = df_frame[df_frame.nflId == ballCarrierId][["x_clean","y_clean"]].values[0]
    pos_p = tackle_data[["x_clean","y_clean"]].values[0]
    theta_rel = np.arctan2(pos_bc[1] - pos_p[1], pos_bc[0] - pos_p[0])*180/np.pi
    theta_rel = (theta_rel + 360) % 360
    p_heading = tackle_data.dir_clean.values[0]
    # print("BC Angle: {}, Heading: {}".format(theta_rel,p_heading))
    angle_diff = np.abs(p_heading - theta_rel)
    theta_p = 360 - angle_diff if angle_diff > 180 else angle_diff
    return s_p, theta_p

df_opt = restore_geometry(df_opt)

bc_data = df_opt[df_opt.ballCarrierId == df_opt.nflId]
tackle_data = df_opt[df_opt.defensiveTeam == df_opt.club]

def calc_all_pursuit_metrics(tackle_data: pd.DataFrame, bc_data: pd.DataFrame) -> tuple:
    s_p = tackle_data.s_clean
    pos_p = tackle_data[["x_clean","y_clean"]].values
    pos_bc = bc_data[(bc_data.gameId == tackle_data.gameId) & 
                     (bc_data.playId == tackle_data.playId) & 
                     (bc_data.ballCarrierId == bc_data.nflId) &
                     (bc_data.frameId == tackle_data.frameId)][["x_clean","y_clean"]].values[0]
    theta_rel = np.arctan2(pos_bc[1] - pos_p[1], pos_bc[0] - pos_p[0])*180/np.pi
    theta_rel = (theta_rel + 360) % 360
    p_heading = tackle_data.dir_clean
    angle_diff = np.abs(p_heading - theta_rel)
    theta_p = 360 - angle_diff if angle_diff > 180 else angle_diff
    return s_p, theta_p

metrics_precalculated = True

if not metrics_precalculated:
    tqdm.pandas(desc="Calculating Pursuit Metrics")
    tackle_data["s_p"],tackle_data["theta_p"] = zip(*tackle_data.progress_apply(
        lambda x: calc_all_pursuit_metrics(x, bc_data), axis=1)) # type: ignore
    with open("./data/test_pursuit_data.pkl", "wb") as f: 
        pickle.dump(tackle_data, f)
else:
    tackle_data = pickle.load(open(os.path.join(root_dir,"data/test_pursuit_data.pkl"),"rb"))

# 3. Match Player Names to IDs, analyze data -----------------------------------
pursuit_profiles = tackle_data[["nflId","s_p","theta_p"]].groupby("nflId").mean()
pursuit_profiles = pursuit_profiles.round({"s_p":2, "theta_p":2})

df_players = pd.read_csv("./data/players.csv", usecols=["nflId","position","displayName"])
labeled_data = pursuit_profiles.merge(df_players, on="nflId")
labeled_data.sort_values(by=["s_p","theta_p"], ascending=False, inplace=True)

df_tackle = pd.read_csv("./data/tackles.csv", usecols=["nflId","tackle","pff_missedTackle"])
player_tackles = df_tackle.groupby("nflId").sum()
labeled_data = labeled_data.merge(player_tackles, on="nflId")

labeled_data["tackle_rate"] = labeled_data.tackle / (labeled_data.tackle + labeled_data.pff_missedTackle)
labeled_data.dropna(inplace=True)

fig = px.scatter(labeled_data, x="s_p", y="theta_p", color="position", 
                 hover_data=["displayName","tackle_rate"], size="tackle",
                 labels={"s_p":"Pursuit Speed (yds/sec)",
                        "theta_p":"Pursuit Angle (deg)"})
fig.show()