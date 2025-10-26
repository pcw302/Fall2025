import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


CANDY_FILEPATH = 'Data VIs/Candy Data.csv'


candy_df = pd.read_csv(CANDY_FILEPATH)

INDEX_NAMES = {
    'chocolate': 'Chocolate',
    'neither': 'Neither',
    'fruity': 'Fruity',
    'both': 'Both'
}


palette = {'bar': 'darkorange', 'neither': 'indigo', 'pluribus': 'seagreen'}

fig, ax = plt.subplots(figsize=(10, 6))
sns.swarmplot(data=candy_df, x='winpercent',y='is_chocolate', hue='is_pluribus', palette=palette, dodge=True)

#Set axis
ax.set_xlabel('Win Percent', fontsize=12)
ax.set_ylabel('', fontsize=12)
ax.set_title('Win Percent by Flavor Group', fontsize=14, pad=12)
ax.set_yticklabels(INDEX_NAMES.values(), fontsize=12)
ax.set_xlim(0,100)

#Adjust Legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles=handles, labels=[l.title() for l in labels], title='Candy Form',
                   bbox_to_anchor=(1.15, 0.5), loc='upper right', frameon=True, fontsize=10, title_fontsize=11)

#Finding the top performers
def get_highest_performer(df, category):

    df_cat = df[df['is_chocolate'] == category]
    max_idx = df_cat['winpercent'].idxmax()
    max_row = df_cat.loc[max_idx]
    candy_name = max_row['competitorname']
    win_percent = max_row['winpercent']

    return candy_name, win_percent

cats = ['chocolate', 'neither','fruity', 'both']
results = {}
for cat in cats:
    candy, percent = get_highest_performer(candy_df, cat)
    results[cat] = (candy, percent)

#Drawing Arrows

#We need the exact y cords of the points, bc they are not mapped to 0,1,2,3 bc that would be too easy
y_labels_order = [t.get_text() for t in ax.get_yticklabels()]

arrow_props = dict(facecolor='black', arrowstyle='->', lw=1.5, mutation_scale=15)

for cat, (candy, percent) in results.items():
    
    y_coord_index = y_labels_order.index(INDEX_NAMES[cat])
    
    #This is specifically for the starburrst point
    #I have absolutely no idea why the arrow was always off but this 
    #Is the lazy way to fix it 
    if cat == 'fruity':
        y_coord_index += 0.265 
 
 
    xy_coord = (percent, y_coord_index) 
    
    xytext_coord = (percent + 5, y_coord_index - 0.1) 
    
    
    
    
    ax.annotate(f'{candy} ({percent:.1f}%)',
                xy=xy_coord, 
                xytext=xytext_coord, 
                xycoords='data',      
                textcoords='data',    
                arrowprops=arrow_props,
                fontsize=10,
                horizontalalignment='left',
                verticalalignment='center')
#Set Grid
sns.set_style('whitegrid')
ax.xaxis.grid(True, linestyle='--', linewidth=0.6, color='gray', alpha=0.6)
ax.yaxis.grid(False)

#I looked up how to draw horzatal lines bc it was hard to tell where the groups started and ended
for i in range(3):
    ax.hlines(i+0.5, 0, 100, colors='lightgray', linewidth=0.8, zorder=2)

plt.savefig('swarmplot.png', bbox_inches='tight', facecolor='white')
plt.show()
