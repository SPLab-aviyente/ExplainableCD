# ExplainableCD - A python package for explainability of modularity maximization
# Copyright (C) 2022 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA

#####

# This script analyzes network of network scientist using Q_CM in order to show 
# the effect of varying resolution parameter on community structure. 

import pickle

from pathlib import Path

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import project_path
from src import modularity, consensus_clustering

PROJECT_DIR = Path(__file__).parents[1]

####

# Load the graph
file_name = str(Path(PROJECT_DIR, "data", "inputs", "netscience", "netscience.gml"))
G = ig.read(file_name)
G = G.connected_components().giant()

# Find community structures with different resolution parameters
# Since generating communities take a while, found communities will be saved.

time_scales = np.arange(1, 100, 0.2) # 1/res

cs_file = Path(PROJECT_DIR, "data", "outputs", "netscience", "communities.pickle")
if cs_file.exists():
    with open(cs_file, "rb") as f:
        C_all = pickle.load(f)
else:
    C_all = []
    for r, time in enumerate(time_scales):
        
        if (r+1)%50 == 0:
            print("{:d}% is done.".format(int((r+1)/len(time_scales)*100)))

        algo = lambda G: modularity.find_comms(G, res=1/time, n_runs=100)
        partitions = algo(G)
        C_all.append(ig.VertexClustering(
            G, membership=consensus_clustering.find_comms(partitions, algo).astype(int)
        ))
    
    # Save the find communities in case we need them again
    cs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cs_file, "wb") as f:
        pickle.dump(C_all, f)

############
# PLOTTING #
############

sns.set_theme()
sns.set_style("white")
sns.set_context("paper")

# Plot number of communities
n_comms = [int(len(C)) for C in C_all]

fig, ax = plt.subplots()

ax.plot(time_scales, n_comms, 'o', markerfacecolor="#F8CECC", markeredgecolor="#B85450")
ax.set_xscale("log")
ax.grid(axis="y")
ax.set_xlabel(r"Time (1/$\gamma$)")
ax.set_ylabel("Number of Communities")
ax.set_title("Number of communities as a function of time")
ax.tick_params(axis="x", which="both", bottom=True, length=4, width=1)
ax.tick_params(axis="y", left=True, length=4, width=1)

# Save the figure
Path(PROJECT_DIR, "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(Path(PROJECT_DIR, "figures", "network_scientists_ncomms.pdf"), dpi=600, 
            bbox_inches = 'tight', pad_inches = 0.01)

# Plot two different community structures
fig, ax = plt.subplots()

# vertex coordinates
C = C_all[55].membership
weights = [3 if C[e.tuple[0]]==C[e.tuple[1]] else 0.1 for e in G.es]
layout = G.layout_fruchterman_reingold(weights=weights, grid=True)

visual_style = {}

# vertex colors
colors = plt.get_cmap("tab20")
rnd_choice = np.arange(20)
np.random.shuffle(rnd_choice)
visual_style["vertex_color"] = [colors(rnd_choice[comm]) for comm in C_all[0].membership]

markgroups = ["#DAE8FC", "#D5E8D4", "#F8CECC", "#E1D5E7"]

ig.plot(C_all[55], layout=layout, target=ax, mark_groups=True, vertex_size=0, 
        vertex_label=None, edge_width=0)

ig.plot(C_all[0], layout=layout, target=ax, **visual_style, vertex_size=0.25,
        vertex_label=None, edge_width=0.5, edge_color="black")

# Save figure
plt.savefig(Path(PROJECT_DIR, "figures", "network_scientists_comm_struct.pdf"), 
            dpi=600, bbox_inches = 'tight', pad_inches = 0.01)