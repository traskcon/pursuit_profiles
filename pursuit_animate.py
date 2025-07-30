"""
SOURCE: https://github.com/mpchang/uncovering-missed-tackle-opportunities/blob/main/code/plotting/animate_play.py

A script that animates tracking data, given gameId and playId. 
Players can be identified by mousing over the individuals dots. 
The play description is also displayed at the bottom of the plot, 
together with play information at the top of the plot. 

Data should be stored in a dir named data, in the same dir as this script. 

Original Source: https://www.kaggle.com/code/huntingdata11/animated-and-interactive-nfl-plays-in-plotly/notebook
"""

import plotly.graph_objects as go
import plotly.io as pio
import pickle
import os
import numpy as np
from sandbox import calculate_pursuit_metrics
from data_preprocessing import restore_geometry
from constants import CLUB_DICT

root_dir = os.getcwd()

pio.renderers.default = (
    "browser"  # modify this to plot on something else besides browser
)

# Modify the variables below to plot your desired play
game_id = 2022091200
play_id = 3826

# Test cases:
# 2022090800, 1385 = going left, on attacking half

# team colors to distinguish between players on plots
colors = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#241773",
    "BUF": "#00338D",
    "CAR": "#0085CA",
    "CHI": "#C83803",
    "CIN": "#FB4F14",
    "CLE": "#311D00",
    "DAL": "#003594",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#203731",
    "HOU": "#03202F",
    "IND": "#002C5F",
    "JAX": "#9F792C",
    "KC": "#E31837",
    "LA": "#FFA300",
    "LAC": "#0080C6",
    "LV": "#000000",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#002244",
    "NO": "#D3BC8D",
    "NYG": "#0B2265",
    "NYJ": "#125740",
    "PHI": "#004C54",
    "PIT": "#FFB612",
    "SEA": "#69BE28",
    "SF": "#AA0000",
    "TB": "#D50A0A",
    "TEN": "#4B92DB",
    "WAS": "#5A1414",
    "football": "#CBB67C",
    "tackle": "#FFC0CB",
}

# Handle Data I/O
df_opt = pickle.load(open(os.path.join(root_dir,"data/test_tracking_data.pkl"),"rb"))
df_focused = df_opt[(df_opt["playId"] == play_id) & (df_opt["gameId"] == game_id)]
df_focused = restore_geometry(df_focused)

ballcarrierId = df_opt.ballCarrierId.values[0]
tackle_frame_id = -1

# initialize plotly play and pause buttons for animation
updatemenus_dict = [
    {
        "buttons": [
            {
                "args": [
                    None,
                    {
                        "frame": {"duration": 100, "redraw": False},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }
]

# initialize plotly slider to show frame position in animation
sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Frame:",
        "visible": True,
        "xanchor": "right",
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": [],
}

# Frame Info
sorted_frame_list = df_focused.frameId.unique()
sorted_frame_list.sort()

frames = []
for frameId in sorted_frame_list:
    data = []
    # Add Yardline Numbers to Field
    data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    data.append(
        go.Scatter(
            x=np.arange(20, 110, 10),
            y=[53.5 - 5] * len(np.arange(20, 110, 10)),
            mode="text",
            text=list(
                map(str, list(np.arange(20, 61, 10) - 10) + list(np.arange(40, 9, -10)))
            ),
            textfont_size=30,
            textfont_family="Courier New, monospace",
            textfont_color="#ffffff",
            showlegend=False,
            hoverinfo="none",
        )
    )
    # Plot Players
    for club in df_focused.club.unique():
        plot_df = df_focused[
            (df_focused.club == club) & (df_focused.frameId == frameId)
        ].copy()
        # 32 is the clean version of "football"
        if club != 32:
            hover_text_array = []
            for nflId in plot_df.nflId:
                selected_player_df = plot_df[plot_df.nflId == nflId]
                hover_text_array.append(
                    f"nflId:{selected_player_df['nflId'].values[0]}"
                )
                if nflId == 42827:
                    frame_data = df_focused[df_focused.frameId == frameId]
                    v_p, theta_p = calculate_pursuit_metrics(frame_data, nflId)
                    x_p, y_p = selected_player_df[["x_clean","y_clean"]].values[0]
                    p_heading = selected_player_df.dir_clean.values[0]
                    data.append(
                        go.Scatter(
                            x=[x_p,x_p+v_p*np.cos(p_heading*np.pi/180)],
                            y=[y_p,y_p+v_p*np.sin(p_heading*np.pi/180)],
                            marker=dict(size=10,symbol="arrow-bar-up",angleref="previous")
                        )
                    )
            data.append(
                go.Scatter(
                    x=plot_df["x_clean"],
                    y=plot_df["y_clean"],
                    mode="markers",
                    marker_color=colors[next(key for key, value in CLUB_DICT.items()
                                             if value == club)],
                    marker_size=10,
                    name=next(key for key, value in CLUB_DICT.items()
                                             if value == club),
                    hovertext=hover_text_array,
                    hoverinfo="text",
                )
            )
            if (
                plot_df.event.values[0] == "tackle"
                and club == plot_df.possessionTeam.values[0]
            ):
                tackle_frame_id = frameId
                ballcarrier_df = df_focused[
                    (df_focused.nflId == ballcarrierId)
                    & (df_focused.frameId == frameId)
                ].copy()
                data.append(
                    go.Scatter(
                        x=ballcarrier_df["x_clean"],
                        y=ballcarrier_df["y_clean"],
                        mode="markers",
                        marker_color=colors["tackle"],
                        marker_size=25,
                        name="tackle",
                        hovertext=["Tackle"],
                        hoverinfo="text",
                    )
                )
        else:
            data.append(
                go.Scatter(
                    x=plot_df["x_clean"],
                    y=plot_df["y_clean"],
                    mode="markers",
                    marker_color=colors[next(key for key, value in CLUB_DICT.items()
                                             if value == club)],
                    marker_size=10,
                    name=next(key for key, value in CLUB_DICT.items()
                                             if value == club),
                    hoverinfo="none",
                )
            )

    # add frame to slider
    slider_step = {
        "args": [
            [frameId],
            {
                "frame": {"duration": 100, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
        "label": str(frameId),
        "method": "animate",
    }
    sliders_dict["steps"].append(slider_step)
    frames.append(go.Frame(data=data, name=str(frameId)))

scale = 10
layout = go.Layout(
    autosize=False,
    width=120 * scale,
    height=60 * scale,
    xaxis=dict(
        range=[0, 120],
        autorange=False,
        tickmode="array",
        tickvals=np.arange(10, 111, 5).tolist(),
        showticklabels=False,
    ),
    yaxis=dict(range=[0, 53.3], autorange=False, showgrid=False, showticklabels=False),
    plot_bgcolor="#00B140",
    updatemenus=updatemenus_dict,
    sliders=[sliders_dict],
)

fig = go.Figure(data=frames[0]["data"], layout=layout, frames=frames[1:])

fig.show()