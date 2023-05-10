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

# This script analyzes US airport network using Q_CM and resolution parameter 
# set to 1. The purpose is to observe different roles that can be assigned to 
# nodes based on the community structure of the netwokr.

from pathlib import Path

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import project_path
from src import modularity, consensus_clustering, node_roles

PROJECT_DIR = Path(__file__).parents[1]

# Load graph
file_name = str(Path(PROJECT_DIR, "data", "inputs", "naairportnet", 
                     "uscities.gml"))
G = ig.read(file_name)

# Find community structure
algo = lambda G: modularity.find_comms(G, n_runs=100)
C = ig.VertexClustering(
    G, membership=consensus_clustering.find_comms(algo(G), algo).astype(int)
)

# Calculate metrics for node roles
zscore = node_roles.module_degree_zscore(C)
pcoeff = node_roles.participation_coeff(C)

############
# PLOTTING #
############

sns.set_theme()
sns.set_style("white")
sns.set_context("paper")

# Plot participation coeff and z-scores
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.scatter(pcoeff, zscore, facecolor="#F8CECC", edgecolor="#B85450")

# draw role assingment regions  
ax.axhline(2.5, color="black", linestyle="--")
ax.vlines(0.05, -1.5, 2.5, color="black", linestyle="--")
ax.vlines(0.62, -1.5, 2.5, color="black", linestyle="--")
ax.vlines(0.8, -1.5, 2.5, color="black", linestyle="--")
ax.vlines(0.3, 2.5, 8, color="black", linestyle="--")
ax.vlines(0.75, 2.5, 8, color="black", linestyle="--")
ax.annotate("R1", (-0.05, 1.8), ha="center", va="center")
ax.annotate("R2", (0.075, 1.8), ha="left", va="center")
ax.annotate("R3", (0.65, 1.8), ha="center", va="center")
ax.annotate("R4", (0.85, 1.8), ha="center", va="center")
ax.annotate("R5", (-0.05, 7.3), ha="center", va="center")
ax.annotate("R6", (0.35, 7.3), ha="center", va="center")
ax.annotate("R7", (0.8, 7.3), ha="center", va="center")

# axes setting
ax.set_ylim(-1.5, 8)
ax.set_xlim(-0.1, 1.0)
ax.set_xlabel("Participation Coefficient")
ax.set_ylabel("Within Module Z-Score")
ax.set_title("Participation Coefficient vs Within Module Z-Score")
ax.tick_params(axis="x", bottom=True, length=4, width=1)
ax.tick_params(axis="y", left=True, length=4, width=1)

# save the figure
Path(PROJECT_DIR, "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(Path(PROJECT_DIR, "figures", "usairport_pcoeff_vs_zscore.pdf"), 
            dpi=600, bbox_inches = 'tight', pad_inches = 0.01)

# Plot the cities with connector hubs indicated
connector_hubs = np.nonzero(np.logical_and(zscore>2.5, pcoeff>0.3, pcoeff<0.75))[0]
layout = [[v["X"], v["Y"]] for v in G.vs]
layout = ig.Layout(coords=layout)

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.get_cmap("tab20")
ig.plot(C, target=ax, layout=layout, edge_width=0,
        vertex_color=[colors(c) for c in C.membership],
        vertex_size=[6000 if i in connector_hubs else 3000 for i in range(G.vcount())])

# save the figure
Path(PROJECT_DIR, "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(Path(PROJECT_DIR, "figures", "usairport_cities.pdf"), 
            dpi=600, bbox_inches = 'tight', pad_inches = 0.01)