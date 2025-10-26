import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


CANDY_FILEPATH = 'Data VIs/Candy Data.csv'


candy_df = pd.read_csv(CANDY_FILEPATH)

tootsie_pop_indx = candy_df[candy_df['competitorname'] == 'Tootsie Pop'].index

altered_candy_df = candy_df.drop(tootsie_pop_indx)

#Fixing Grammer for the catagories and legend ( The incorrect grammer is bothering me)
INDEX_NAMES = {
    'neither': 'Neither',
    'chocolate': 'Chocolate',
    'fruity': 'Fruity'
}

DISPLAY_NAMES = {
    'often': 'Often',
    'sometimes': 'Sometimes',
    'rarely': 'Rarely',
    'a_lot': 'A lot',
    'neither': 'Neither'
}

COLORS = ['DarkOrange', 'DarkViolet', 'ForestGreen', 'FireBrick']
#Relevent Columns : win_cat , is_chocolate

raw_table = pd.crosstab(altered_candy_df['is_chocolate'], altered_candy_df['win_cat'])
raw_table = raw_table.rename(index = INDEX_NAMES, columns = DISPLAY_NAMES)

sums = raw_table.sum(axis = 1)
perc = raw_table.div(sums, axis = 0) * 100


ax = perc.plot.bar(stacked = True, color =COLORS, figsize = [6,6], 
                    rot = 0, fontsize = 16, alpha = 0.9, ec = 'whitesmoke')
[ax.spines[i].set_visible(False) for i in ax.spines]
ax.get_yaxis().set_visible(False)
ax.tick_params(length = 0)
plt.xlabel("Win Count (n=84)", fontsize=16)
plt.title("What Type of Halloween Candy Wins?", fontsize=18, pad=14)
plt.legend(bbox_to_anchor = [1,1]) #just moving it out of the way

sums = raw_table.sum(axis = 1)
perc = raw_table.div(sums, axis = 0) * 100

#Make percentages
for patch in ax.patches:
    x = patch.get_x()
    y = patch.get_y()
    width = patch.get_width()
    height = patch.get_height()
    
    cx = x + width/2
    if height >= 5:
        label = f"{height:.0f}%"
    else:
        label = f"{height:.0f}%"
    text_color = 'white' if height > 15 else 'black'
    if height < 4:
        va = 'bottom'
        y_text = y + height + 1
        fontsize = 9
    else:
        va = 'center'
        y_text = y + height/2
        fontsize = 10
    ax.text(cx, y_text, label, ha='center', va=va, color=text_color, fontsize=fontsize, weight='bold', clip_on=False)


group_sizes = raw_table.sum(axis=1)
x_labels = [f"{idx}\n(n={int(group_sizes.iloc[i])})" for i, idx in enumerate(group_sizes.index)]
# Use tick positions that match the number of labels
ticks = range(len(sums))
plt.xticks(ticks, labels = x_labels)
plt.tick_params(pad = 10)


#Get counts for legend
counts_by_win = raw_table.sum(axis=0)

handles, labels = ax.get_legend_handles_labels()

new_labels = [f"{DISPLAY_NAMES.get(lbl, lbl.title())} ({int(counts_by_win[lbl])})" for lbl in labels]

leg = plt.legend(handles = handles[::-1], labels = new_labels[::-1], bbox_to_anchor = [0.9, 0.75], title = "Win Catagory", 
                 title_fontsize = 18, fontsize = 14, markerfirst = False, frameon = False) 
leg.get_title().set_multialignment('center')

fig = ax.get_figure()

#Text box
fig.text(0.9, 0.3, 'Tootsie Roll Pops, which are both Fruity\n and Chocolate, were excluded from\n the figure.', fontsize=9,
         bbox=dict(facecolor='DarkGray', edgecolor='0.7', boxstyle='round,pad=0.6'))

plt.savefig('stacked_bar.png', bbox_inches='tight', facecolor='white')
plt.show() 