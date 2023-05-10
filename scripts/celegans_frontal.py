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

# This script analyzes C. Elegans frontal neural network using Q_ER and Q_CM
# in order to show the difference in community structures found with different 
# null models.

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
file_name = str(Path(PROJECT_DIR, "data", "celegans_frontal", "C-elegans-frontal.txt"))
G = ig.Graph.Read(file_name, directed=False)

# Community detection with Q_ER
algo_er = lambda G: modularity.find_comms(G, res=1, null_model="er", n_runs=100)
C_er = algo_er(G)
C_er = ig.VertexClustering(
    G, membership=consensus_clustering.find_comms(C_er, algo_er).astype(int)
)

# Community detection with Q_CM
algo_er = lambda G: modularity.find_comms(G, res=0.88, null_model="cm", n_runs=100)
C_cm = algo_er(G)
C_cm = ig.VertexClustering(
    G, membership=consensus_clustering.find_comms(C_cm, algo_er).astype(int)
)

############
# PLOTTING #
############

# Get vertex coordinates
d = np.array(G.degree())
layoutg = G.layout_kamada_kawai()

x = np.array([x[0] for x in layoutg.coords])
y = np.log(d)

# normalize coordinates
x -= np.min(x)
x /= np.max(x)

y -= np.min(y)
y /= np.max(y)

layoutd = [[x[i], y[i]]for i in range(len(x))]
layoutd = ig.Layout(coords=layoutd)

colors = plt.get_cmap("tab10") # colors to use for vertices

sns.set_theme()
sns.set_style("white")
sns.set_context("paper")
fig = plt.figure(figsize=(5,5))

# plot Q_ER results
ax = fig.add_axes((0.1, 0.0, 0.45, 1))
ig.plot(C_er, target=ax, layout=layoutd, vertex_size=.05, edge_width=0.1,
        vertex_color=[colors(c) for c in C_er.membership])

ax.spines["left"].set_visible(True)
yticklabels = np.arange(0, np.log(np.max(d)), 0.5)
ax.set_yticks(np.linspace(0, 1, len(yticklabels)), labels=yticklabels)
ax.set_ylabel("Log Degree")
ax.tick_params(axis="y", left=True, length=4, width=1)

# plot Q_CM results
ax = fig.add_axes((0.55, 0.0, 0.45, 1))
ax = ig.plot(C_cm, target=ax, layout=layoutd, vertex_size=.05, edge_width=0.1,
        vertex_color=[colors(c) for c in C_cm.membership])

# Save the figure
Path(PROJECT_DIR, "figures").mkdir(parents=True, exist_ok=True)
plt.savefig(Path(PROJECT_DIR, "figures", "celegans_frontal.pdf"), dpi=600, 
            bbox_inches = 'tight', pad_inches = 0.01)