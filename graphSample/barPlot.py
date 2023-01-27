"""A pretty bar chart from 
https://towardsdatascience.com/5-steps-to-build-beautiful-bar-charts-with-python-3691d434117a
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

df = pd.read_csv('./graphSample/delayedFlights1991.csv')

# Only keep the columns for graph
df = df[['Month', 'ArrDelay']]
# Removed cancelled or diverted flights
df = df[~df['ArrDelay'].isnull()]
# print(df['Month'].unique())

# Group by month and get the mean
delay_by_month = df.groupby(['Month']).mean()['ArrDelay'].reset_index()
# print(delay_by_month.shape)
# print(delay_by_month.head())

#=================
# The basic plot
#=================

# Create the figure  and axes objects, specify the size and the dots per inches
fig, ax = plt.subplots(figsize=(13.33,7.5), dpi=96)
# Plot the bars
bar1 = ax.bar(delay_by_month['Month'], delay_by_month['ArrDelay'], width=0.6)
# plt.show()

#=========================================
# The essentials to add to the bare basic
#=========================================

# Create the grid
ax.grid(which='major', axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
ax.grid(which='major', axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

# Reformat x-axis label and tick labels
ax.set_xlabel('', fontsize=12, labelpad=10) # in this case, don't need x label
ax.xaxis.set_label_position('bottom')
ax.xaxis.set_major_formatter(lambda s, i : f'{s:,.0f}') # format the major ticket
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12, labelrotation=0)
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(delay_by_month['Month'], labels=labels)

# Reformat y-axis
ax.set_ylabel('Delay (minutes)', fontsize=12, labelpad=10)
ax.yaxis.set_label_position('left')
ax.yaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

# Add labels on top of each bar
ax.bar_label(bar1, labels=[f'{e:,.1f}' for e in delay_by_month['ArrDelay']], padding=3, color='black', fontsize=8)

#========================
# The Professional look
#========================

# Remove the spines
ax.spines[['top', 'left', 'bottom']].set_visible(False)

# Make the left spine thicker
ax.spines['right'].set_linewidth(1.1)

# Add in the red line and rectagle on top
ax.plot([0.12, .9], [.98, .98], transform=fig.transFigure, clip_on=False, color='#E3120B', linewidth=.6)
ax.add_patch(plt.Rectangle((0.12, .98), 0.04, -0.02, facecolor='#E3120B', transform=fig.transFigure, 
            clip_on=False, linewidth=0))

# Add in title and subtitle
ax.text(x=0.12, y=.93, s="Average Airlines Delay per Month in 1991", transform=fig.transFigure, ha='left',
        fontsize=14, weight='bold', alpha=.8)
ax.text(x=0.12, y=.90, s="Difference in minutes between scheduled and actual arrival time averaged over each month", 
    transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)

# Set source text
ax.text(x=0.1, y=0.12, s="Source: Kaggle - Airlines Delay -  https://www.kaggle.com/datasets/giovamata/airlinedelaycauses", 
    transform=fig.transFigure, ha='left', fontsize=10, alpha=.7)

# Adjust the margins around the plot area
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)

# Set a white background
fig.patch.set_facecolor('white')

#====================
# Color Gradient
#====================

# Colours - choose the extreme colours of the colour map
colours = ['#2196f3', '#bbdefb']

# Colormap - build the colour maps
cmap = mpl.colors.LinearSegmentedColormap.from_list("color_map", colours, N=256)
# Linearly normalize the data into [0.0, 1.0] interval
norm = mpl.colors.Normalize(delay_by_month['ArrDelay'].min(), delay_by_month['ArrDelay'].max())

# Plot bars
bar1 = ax.bar(delay_by_month['Month'], delay_by_month['ArrDelay'], color=cmap(norm(delay_by_month['ArrDelay'])), 
    width=0.6, zorder=2)

#=======================
# The Finishing touches
#======================

# Find the average data point and split the series in 2
average = delay_by_month['ArrDelay'].mean()
below_average = delay_by_month[delay_by_month['ArrDelay']<average]
above_average = delay_by_month[delay_by_month['ArrDelay']>=average]

# Colours - Choose the extreme colours of the colour map
colors_high = ['#ff5a5f', '#c81d25']
colors_low = ['#2196f3', '#bbdefb']

# Colormap - build the colour maps
cmap_low = mpl.colors.LinearSegmentedColormap.from_list("low_map", colors_low, N=256)
cmap_high = mpl.colors.LinearSegmentedColormap.from_list("high_map", colors_high, N=256)
norm_low = mpl.colors.Normalize(below_average['ArrDelay'].min(), average)
norm_high = mpl.colors.Normalize(average, above_average['ArrDelay'].max())

# Plot bars and average (horizontal) line
bar1 = ax.bar(below_average['Month'], below_average['ArrDelay'], color=cmap_low(norm_low(below_average['ArrDelay'])), 
width=0.6, label='Below Average', zorder=2)
bar2 = ax.bar(above_average['Month'], above_average['ArrDelay'], color=cmap_high(norm_high(above_average['ArrDelay'])), 
width=0.6, label='Above Average', zorder=2)
plt.axhline(y=average, color='grey', linewidth=3)

# Determine the y-limits of the plot
ymin, ymax = ax.get_ylim()
# Calculate a suitable y position for the text label
y_pos = average/ymax + 0.03
# Annotate the average line
ax.text(0.88, y_pos, f'Average = {average:.1f}', ha='right', va='center', transform=ax.transAxes, size=8, zorder=3)

# Add legend
ax.legend(loc='best', ncol=2, bbox_to_anchor=[1, 1.07], borderaxespad=0, frameon=False, fontsize=8)

plt.show()