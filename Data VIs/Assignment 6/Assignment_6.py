import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm


college_df = pd.read_csv('College Data.csv')

top_200 = college_df.iloc[0:200]

top_10 = college_df.iloc[0:10]
#print(top_200.head())




f, axes =  plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'wspace':0.2,
                                                             'hspace':0.3})
axes = axes.flatten()

y = ['total_score']
x = ['num_students', 'student_staff_ratio', 'international_students', 'female']

x_labels = ['Number of Students', 'Student to Staff Ratio', 'International Students (%)', 'Female Students (%)']

for i in range(len(axes)):  
    sns.scatterplot(data = college_df, x = college_df[x[i]], y = college_df[y[0]],
                    s = 70, alpha = 0.8, ec = 'k', ax = axes[i], color='indianred')
    
    sns.scatterplot(data=top_10, x=top_10[x[i]], y=top_10[y[0]],
                    s=70, ec='k', ax=axes[i], color='gold', marker='*', label='Top 10 University')
    
    
    median_x = college_df[x[i]].median()
    axes[i].axvline(median_x, color='darkslategray', linestyle=':', linewidth=4, label='Median', alpha=1)
    
    axes[i].get_legend().remove()
    
    if i == 0 or i == 2:
        axes[i].set_ylabel('Total Score', fontsize = 14, labelpad = 15)
    else:
        axes[i].set_ylabel('')
    axes[i].set_xlabel(x_labels[i], fontsize = 14, labelpad = 10)
    axes[i].tick_params(labelsize = 12)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].set_ylim(0, 100)

handles, labels = axes[0].get_legend_handles_labels()
f.legend(handles, labels, loc='upper center', ncol=1, bbox_to_anchor=(0.2, 1), fontsize=14, facecolor='whitesmoke', )
f.suptitle('Comparison of University Metrics', fontsize=16, y=0.95)
plt.savefig('Scatter.png',bbox_inches = 'tight',facecolor='white')
plt.show()

correlations = {}
for variable in x:
    correlations[variable] = college_df[variable].corr(college_df['total_score'])

for variable, corr in correlations.items():
    print(f"Correlation between {variable} and total_score: {corr:.2f}")
    
    
    
    
bot_200 = college_df.iloc[-200:]
first_last = pd.concat([top_200, bot_200]).reset_index()
first_last['Ranks'] = ['top']*200 +['last']*200

from pandas.plotting import parallel_coordinates as pcp
from matplotlib.patches import Patch

pcp_df = first_last[['teaching', 'research', 'citations', 'international', 'female', 'income',  'Ranks']]

pcp_df.columns = [col.replace('_', ' ').title() for col in pcp_df.columns]


plt.figure(figsize = (12,10))
colors = ['cornflowerblue', 'crimson']
pcp(pcp_df, 'Ranks',  color = colors, alpha = 0.5, axvlines_kwds={'alpha':0.5,'color':'grey'})
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(axis = 'y', alpha = 0.4, linestyle = '--')


handles = [Patch(facecolor=c, alpha = 0.8) for c in colors]
plt.legend(handles, ['Top 200','Bottom 200'], title='Ranks',
           bbox_to_anchor=(0.9,1.15), loc='upper left', fontsize = 14, title_fontsize = 16)
plt.ylabel('Normalized Values', fontsize = 14, labelpad = 20)
plt.savefig('PCP.png',bbox_inches ='tight',facecolor='whitesmoke')
plt.show()

f, axes =  plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={'wspace':0.2,
                                                             'hspace':0.3})
axes = axes.flatten()
y = ['total_score']
x = ['num_students', 'student_staff_ratio', 'international_students', 'female']

x_labels = ['Log of Number of Students', 'Student to Staff Ratio', 'International Students (%)', 'Female Students (%)']


for i in range(len(axes)):
    x_data = college_df[x[i]]
    x_label = x_labels[i]

    if i == 0:  
        x_data = np.log1p(x_data)

    im = axes[i].hexbin(x_data, college_df[y[0]], gridsize=10, cmap='viridis', mincnt=1, norm=LogNorm())

    if i == 0 or i == 2:
        axes[i].set_ylabel('Total Score', fontsize = 14, labelpad = 15)
    else:
        axes[i].set_ylabel('')
    axes[i].set_xlabel(x_label, fontsize = 14, labelpad = 10)
    axes[i].tick_params(labelsize = 12)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    axes[i].set_ylim(0, 100)

# This creates log ticks on the color bar so that it reflects both the log scaled data and the non scaled counts
counts = im.get_array()
min_count, max_count = counts.min(), counts.max()
ticks = np.logspace(np.log10(min_count), np.log10(max_count), num=int(np.log10(max_count))+2).astype(int)
ticks = sorted(list(set(ticks))) 

cbar = f.colorbar(im, ax=axes.ravel().tolist(), label='Count in Bin', ticks=ticks)
cbar.set_ticklabels([str(t) for t in ticks])

f.suptitle('Comparison of University Metrics using Hexbin', fontsize=16, y=0.95)
plt.savefig('hex.png',bbox_inches = 'tight',facecolor='white')
plt.show()