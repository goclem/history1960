from audioop import adpcm2lin
import pandas as pd
from matplotlib import pyplot

def display(image:np.ndarray, title:str='', cmap:str='gray', path:str=None) -> None:
    '''Displays an image'''
    fig, ax = pyplot.subplots(1, figsize=(10, 10))
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    pyplot.tight_layout()
    if path is not None:
        pyplot.savefig(path, dpi=300)
    else:
        pyplot.show()

data = dict(
    year=[1760, 1860, 1960, 2020],
    totalpop=[28.13, 36.75, 44.38, 65.71],
    urbanpop=[6.78, 9.90, 22.57, 37.94],
    ncities=[1622, 979, 337, 382],
    urbansize=[1.3, 1.6, 2.0, 3.4],
    urbandens=[917.8, 1100.8, 2068.9, 2011.9],
)
data = pd.DataFrame(data)

fig, ax = pyplot.subplots(1, figsize=(10, 10))
ax.plot(data.year, np.log(data.totalpop), label='log population')
ax.plot(data.year, np.log(data.urbanpop), label='log urban population')
ax.plot(data.year, np.log(data.ncities),  label='Number of cities')
ax.set_title('Population', fontsize=20)

fig, ax1 = pyplot.subplots(1, figsize=(10, 10))
ax2 = ax1.twinx()
ax1.plot(data.year, data.urbansize.pct_change(), c='red', label='log urban size')
ax2.plot(data.year, data.urbandens.pct_change(), c='blue', label='log urban density')
ax1.set_ylabel('log urban size', color='red')
ax2.set_ylabel('log urban density', color='blue')
ax.set_title('Population', fontsize=20)

fig, ax = pyplot.subplots(1, figsize=(10, 10))
ax.plot(data.year, data.urbansize.pct_change(), c='red', label='log urban size')
ax.plot(data.year, data.urbanpop.pct_change(), c='blue', label='log urban density')




fig, ax1 = plt.subplots()


ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('X data')


plt.show()


fig, ax = pyplot.subplots(1, figsize=(10, 10))

ax.plot(data.year, np.log(data.urbanpop), label='log urban population')
ax.set_title('Population', fontsize=20)


#%% DENSITY PLOT

import numpy as np
from sklearn import neighbors
from matplotlib import patches, pyplot
from numpy import random

# Data
random.seed(1)
locs = np.concatenate((
    random.normal(loc=-7.5, scale=0.5, size=30), 
    random.normal(loc=0,    scale=0.65, size=60), 
    random.normal(loc=6,    scale=0.5, size=10)
    ))
sizes   = random.uniform(5, 10, size=locs.shape[0])
livable = np.concatenate((np.linspace(-10, -5, 50), np.linspace(-4, 8, 120)))
ground  = ([-10, -5, -5, -4, -4, 8, 9, 10], [0, 0, -0.01, -0.01, 0, 0, 0.025, 0.025])
xplot   = np.linspace(-10, 10, 1000)

# Densities
kde  = neighbors.KernelDensity(bandwidth=1.5, kernel='gaussian')
kde.fit(locs.reshape(-1, 1), sample_weight=sizes)
density = np.exp(kde.score_samples(xplot.reshape(-1, 1)))

# Counterfactual densities
densities_c = list()
for i in range(100):
    locs_c = random.choice(livable, len(locs), replace=True).reshape(-1, 1)
    kde.fit(locs_c.reshape(-1, 1), sample_weight=sizes)
    density_c = np.exp(kde.score_samples(xplot.reshape(-1, 1)))
    densities_c.append(density_c)
density_c = np.mean(np.array(densities_c), axis=0)

# Figure density
fig, ax = pyplot.subplots(figsize=(10,6))
# Buildings
ax.scatter(locs, np.repeat(0.005, locs.shape[0]), s=sizes**2, c='red', edgecolor='black', zorder=2, label='Observed buildings')
ax.scatter([-4.5, 9.5], [0.005, 0.030], s=5**2, c='red', edgecolor='black', zorder=2)
ax.plot(xplot, density, color='blue', label='Observed density')
# Unbuildable
ax.axvspan(-5, -4, alpha=0.25, color='red', label='Non buildable')
ax.axvspan(8, 10,  alpha=0.25, color='red')
# Ground
ax.plot(ground[0], ground[1], color='gray')
ax.fill_between(ground[0], ground[1], -0.015, color='lightgray', label='Ground')
# Graphical parameters
ax.set_xlim(-10, 10)
ax.set_ylim(-0.015, 0.2)
ax.set_xlabel('Space', fontsize=20, labelpad=10)
ax.set_ylabel('Density', fontsize=20, labelpad=10)
ax.set_xticks(np.arange(-10, 10), np.arange(-10, 10))
ax.grid(axis='x', linewidth=0.5)
ax.tick_params(labelbottom=False, labelleft=False, colors='white')
# Legend
handles, labels = ax.get_legend_handles_labels()
patch = patches.Patch(color='white', label='Pixel')
patch.set_edgecolor('lightgray')
handles.append(patch)
ax.legend(handles=handles, loc='upper right', edgecolor='white')
pyplot.tight_layout()
pyplot.savefig('/Users/clementgorin/Dropbox/research/seminar_uqam/figures/fig_delineation_density.pdf', dpi=300, transparent=True)

# Figure counterfactual
fig, ax = pyplot.subplots(figsize=(10,6))
# Buildings
ax.plot(xplot, densities_c[0], color='purple', linestyle='--', label='Bootstrapped densities')
for i in np.arange(1, 10):
    ax.plot(xplot, densities_c[i], color='purple', linestyle='--')
ax.scatter(locs_c, np.repeat(0.005, locs.shape[0]), s=sizes**2, c='red', edgecolor='black', zorder=2, label='Bootstrapped buildings')
# Unbuildable
ax.axvspan(-5, -4, alpha=0.25, color='red')
ax.axvspan(8, 10,  alpha=0.25, color='red')
# Ground
ax.plot(ground[0], ground[1], color='gray')
ax.fill_between(ground[0], ground[1], -0.015, color='lightgray')
# Graphical parameters
ax.set_xlim(-10, 10)
ax.set_ylim(-0.015, 0.2)
ax.set_xlabel('Space', fontsize=20, labelpad=10)
ax.set_ylabel('Density', fontsize=20, labelpad=10)
ax.set_xticks(np.arange(-10, 10), np.arange(-10, 10))
ax.grid(axis='x', linewidth=0.5)
ax.tick_params(labelbottom=False, labelleft=False, colors='white')
ax.legend(loc='upper right', edgecolor='white')
pyplot.tight_layout()
pyplot.savefig('/Users/clementgorin/Dropbox/research/seminar_uqam/figures/fig_delineation_counterfactual.pdf', dpi=300, transparent=True)

# Figure Compare
fig, ax = pyplot.subplots(figsize=(10,6))
# Buildings
ax.plot(xplot, density, color='blue', label='Observed density')
ax.plot(xplot, density_c, color='purple', label='Counterfactual density')
ax.scatter(locs, np.repeat(0.005, locs.shape[0]), s=sizes**2, c='red', edgecolor='black', zorder=2)
# Unbuildable
ax.axvspan(-9, -6, alpha=0.25, color='lime', label='Urban areas')
ax.axvspan(-2, 2,  alpha=0.25, color='lime')
ax.axvspan(-5, -4, alpha=0.25, color='red')
ax.axvspan(8, 10,  alpha=0.25, color='red')
# Ground
ax.plot(ground[0], ground[1], color='gray')
ax.fill_between(ground[0], ground[1], -0.015, color='lightgray')
# Graphical parameters
ax.set_xlim(-10, 10)
ax.set_ylim(-0.015, 0.2)
ax.set_xlabel('Space', fontsize=20, labelpad=10)
ax.set_ylabel('Density', fontsize=20, labelpad=10)
ax.set_xticks(np.arange(-10, 10), np.arange(-10, 10))
ax.grid(axis='x', linewidth=0.5)
ax.tick_params(labelbottom=False, labelleft=False, colors='white')
ax.legend(loc='upper right', edgecolor='white')
pyplot.tight_layout()
pyplot.savefig('/Users/clementgorin/Dropbox/research/seminar_uqam/figures/fig_delineation_comparison.pdf', dpi=300, transparent=True)

# %%
