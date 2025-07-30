"""
Exploration of Tackling data, attempting to classify tackle opportunities, missed tackle opportunities, and total tackles.
Research Question: Is being within 1 yard of the ballcarrier for 0.5s sufficient to count as a tackle opportunity?
    * Euclidean distance was the most important factor in "Uncovering Missed Tackle Opportunities", 
        which used an XGBoost classifier to generate tackle probabilities based on 9 features
    * If we are primarily concerned with open-field tackles, the consideration of blocker influence, team influence, 
        and voronoi area are likely insignificant (and play type was already shown to be irrelevant)
    * Relative speed and angle are important, but these are our features of interest, 
        so we don't need to consider them as factors

Proposed Definitions:
    * Tackle Opportunity: When defender X is within 1 yard of the ballcarrier for a > 0.5 second interval.
    * Missed Tackle Opportunity: When the distance between defender X and the ballcarrier is < 1 yard for a > 0.5 second interval,
        then increases back above 1 yard
    * Converted Tackle Opportunity: When defender X is assigned a tackle in the tackle data
        * NOTE: TO != MTO + CTO, as the tackle data may not credit a defender for a tackle made by their teammates, 
            despite remaining within 1 yard of the ballcarrier (and therefore not missing the tackle opportunity)
    * Tackle Opportunity Rate: Tackle Opportunities / Total Active Plays
    * Missed Opportunity Rate: Missed Tackle Opportunities / Tackle Opportunities
    * Tackle Conversion Rate: Tackles Made or Assisted / Tackle Opportunities
        * NOTE: MOR + TCR != 1, as the tackle data may not credit a defender for a tackle made by their teammates
"""

import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import sys
import re
from itertools import groupby

from data_preprocessing import restore_geometry

# Suppress set with copy warning
pd.options.mode.chained_assignment = None

root_dir = os.getcwd()

tackle_df = pd.read_csv("./data/tackles.csv")
df_opt = pickle.load(open(os.path.join(root_dir,"data/test_tracking_data.pkl"),"rb"))
df_opt = restore_geometry(df_opt)

bc_data = df_opt[df_opt.ballCarrierId == df_opt.nflId]
tackle_data = df_opt[df_opt.defensiveTeam == df_opt.club]

# Each frame is 0.1s
def calculate_ballcarrier_distance(tackle_data, bc_data):
    pos_p = tackle_data[["x_clean","y_clean"]].values
    pos_bc = bc_data[(bc_data.gameId == tackle_data.gameId) & 
                     (bc_data.playId == tackle_data.playId) & 
                     (bc_data.ballCarrierId == bc_data.nflId) &
                     (bc_data.frameId == tackle_data.frameId)][["x_clean","y_clean"]].values[0]
    return np.linalg.norm(pos_p - pos_bc)

tqdm.pandas(desc="Calculating Tackle Opportunities")
tackle_data["bc_dist"] = tackle_data.progress_apply(
        lambda x: calculate_ballcarrier_distance(x, bc_data), axis=1) #type: ignore

tackle_data["tackle_opp_check"] = np.where(tackle_data.bc_dist <= 1, 1, 0)
tackle_data["tackle_opp"] = 0
tackle_data["missed_tackle_opp"] = 0

'''ind = prev = 0
tackle_data["tackle_opp"] = 0
for k, v in groupby(tackle_data["tackle_opp_check"], key=lambda x: x == 1):
    ind += sum(1 for _ in v)
    if k and ((ind - prev) >= 5):
        # If ther are more than 5 1s in a row (tackler within 1 yard of bc for 0.5 seconds)
        tackle_data["tackle_opp"].iloc[prev+5] = 1
        # Set a flag in the series'''

opp_string = tackle_data.tackle_opp_check.astype(str).str.cat() #type: ignore
to = 5 * "1" # Substring for a tackle opportunity (5 seconds of within 1 yard)
mto = to + "0" # Substring for a missed tackle opportunity (to followed by moving outside 1 yard)
to_idx = [match.end() for match in re.finditer(to, opp_string)]
mto_idx = [match.end() for match in re.finditer(mto, opp_string)]

tackle_data.iloc[to_idx,-2] = 1
tackle_data.iloc[mto_idx, -1] = 1

# Check tackle opportunity counts
print(tackle_data[["nflId","tackle_opp","missed_tackle_opp"]].groupby("nflId").sum())