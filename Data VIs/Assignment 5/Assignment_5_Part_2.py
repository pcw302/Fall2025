import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


CANDY_FILEPATH = 'Data VIs/Candy Data.csv'


candy_df = pd.read_csv(CANDY_FILEPATH)

#winpercent 
#pricepercent 
#is_chocolate


relevent_candy_df = candy_df[['winpercent', 'pricepercent', 'is_chocolate']]

# ensure numeric columns are numeric
for col in ['winpercent', 'pricepercent']:
    relevent_candy_df[col] = pd.to_numeric(relevent_candy_df[col], errors='coerce')

# diagnostic prints (remove when satisfied)
print("is_chocolate values:", relevent_candy_df['is_chocolate'].value_counts(dropna=False).to_dict())
print("Numeric summary:\n", relevent_candy_df[['winpercent','pricepercent']].describe())

# correct filtering: match the string 'chocolate' (case-insensitive)
chocolate_candy_df = relevent_candy_df[relevent_candy_df['is_chocolate'].astype(str).str.lower() == 'chocolate'].dropna(subset=['winpercent','pricepercent'])
non_chocolate_candy_df = relevent_candy_df[relevent_candy_df['is_chocolate'].astype(str).str.lower() != 'chocolate'].dropna(subset=['winpercent','pricepercent'])

print("Chocolate rows:", len(chocolate_candy_df), "Non-chocolate rows:", len(non_chocolate_candy_df))





COLORS = ['DarkOrange', 'DarkViolet', 'ForestGreen', 'FireBrick']

# Hist 1 is chocolate winpercent
# Hist 2 is Chocolate pricepercent
# Hist 3 is Non-Chocolate winpercent
# Hist 4 is Non-Chocolate pricepercent

# Constants and params
LABEL_SIZE = 10
TICK_SIZE = 7
n_bins_winpercent = 8
n_bins_pricepercent= 15

f, axes =  plt.subplots(2, 2, figsize=(12, 10))
ax_hist_1, ax_hist_2, ax_hist_3, ax_hist_4 = axes.ravel()

# First Hist, Chocalate Winpercent
ax_hist_1.hist(chocolate_candy_df['winpercent'], rwidth = 0.85, bins = n_bins_winpercent, color = COLORS[0])
[ax_hist_1.spines[i].set_visible(False) for i in ax_hist_1.spines]
ax_hist_1.tick_params(labelsize = TICK_SIZE)
ax_hist_1.grid(axis = 'y',color= 'white')
ax_hist_1.set_xlim(0, 100)
ax_hist_1.set_ylim(0, 15)
ax_hist_1.set_ylabel('Count', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_1.set_xlabel('', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_1.set_title('Chocolate Win-Percentage', fontsize = LABEL_SIZE)

# Second Hist, Chocalate Pricepercent
ax_hist_2.hist(chocolate_candy_df['pricepercent'], rwidth = 0.85, bins = n_bins_pricepercent, color = COLORS[1])
[ax_hist_2.spines[i].set_visible(False) for i in ax_hist_2.spines]
ax_hist_2.tick_params(labelsize = TICK_SIZE)
ax_hist_2.grid(axis = 'y',color= 'white')
ax_hist_2.set_xlim(0, 100)
ax_hist_2.set_ylim(0, 15)
ax_hist_2.set_ylabel('Count', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_2.set_xlabel('', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_2.set_title('Chocolate Price-Percentage', fontsize = LABEL_SIZE)

#Third Hist, Non-Chocalate Winpercent
ax_hist_3.hist(non_chocolate_candy_df['winpercent'], rwidth = 0.85, bins = n_bins_winpercent, color = COLORS[2])
[ax_hist_3.spines[i].set_visible(False) for i in ax_hist_3.spines]
ax_hist_3.tick_params(labelsize = TICK_SIZE)
ax_hist_3.grid(axis = 'y',color= 'white')
ax_hist_3.set_xlim(0, 100)
ax_hist_3.set_ylim(0, 15)
ax_hist_3.set_ylabel('Count', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_3.set_xlabel('', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_3.set_title('Non-Chocolate Win-Percentage', fontsize = LABEL_SIZE)

# Fourth Hist, Non-Chocalate Pricepercent
ax_hist_4.hist(non_chocolate_candy_df['pricepercent'], rwidth = 0.85, bins = n_bins_pricepercent, color = COLORS[3])
[ax_hist_4.spines[i].set_visible(False) for i in ax_hist_4.spines]
ax_hist_4.tick_params(labelsize = TICK_SIZE)
ax_hist_4.grid(axis = 'y',color= 'white')
ax_hist_4.set_xlim(0, 100)
ax_hist_4.set_ylim(0, 15)
ax_hist_4.set_ylabel('Count', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_4.set_xlabel('', fontsize = LABEL_SIZE, labelpad = 10)
ax_hist_4.set_title('Non-Chocolate Price-Percentage', fontsize = LABEL_SIZE)

#Adding a boarder around each graph bc I think it looks really ugly without it
for ax in [ax_hist_1, ax_hist_2, ax_hist_3, ax_hist_4]:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('dimgray')
        spine.set_linewidth(0.8)
    ax.patch.set_facecolor('white')
    ax.patch.set_edgecolor('dimgray')
    ax.patch.set_linewidth(0.6)


f.suptitle('Distributions of Win-Percentage and Price-Percentage\nby Chocolate vs Non‑Chocolate', fontsize=13)
plt.savefig('subplots.png.png', bbox_inches='tight', facecolor='white')
plt.show()