import numpy as np, pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os

#### Podíl stavu vozovky na vazných nehodách ####
#### AUTOR: Klára Martinásková

# Načtení dat
df = pd.read_pickle('accidents.pkl.gz')

#Zde si vytvorime jednoduchy graf zobrazujici podil stavu vozovky na vaznych nehodahc
'''''
p16 Stav vozovky: 
1	povrch suchý - neznečištěný
2	povrch suchý - znečištěný (písek, prach, listí, štěrk atd.)
3	povrch mokrý
4	na vozovce je bláto
5	na vozovce je náledí, ujetý sníh - posypané
6	na vozovce je náledí, ujetý sníh  - neposypané
7	na vozovce je rozlitý olej, nafta apod. 
8	souvislá sněhová vrstva, rozbředlý sníh
9	náhlá změna stavu vozovky - (námraza na mostu, místní náledí)
0	jiný stav povrchu vozovky v době nehody

p9 Charakter nehody:
1	nehoda	s následky na životě
2	nehoda	pouze s hmotnou škodou

'''
# pomocna promenna
road_labels = ['jiný stav povrchu vozovky','povrch suchý','povrch suchý - znečištěný','povrch mokrý','bláto','náledí','náledí - neposypané','olej','sněhová vrstva','změna stavu vozovky']

# Hodnoty pro závěrečné výpočty
dry = df[(df.p16 == 1) & (df.p9 == 1) ].shape[0]
wet = df[(df.p16 == 3) & (df.p9 == 1) ].shape[0]
ice = df[(df.p16 == 6) & (df.p9 == 1) ].shape[0]
sum_acc = df.shape[0] #celkovy pocet nehod
acc_life = df[(df.p9 == 1) ].shape[0] # pocet nehod s nasledky na zivot

# Prejmenovani sloupcu
df = df.rename(columns={'p16':'Stav vozovky'})

df.loc[:, 'p9'] = pd.cut(df['p9'], [0, 1, float('inf')], 
                                    labels=['nehoda	s nasledky na zivote', 'nehoda	pouze s hmotnou skodou']) 

df_graphics = df.groupby(['p9', 'Stav vozovky' ]).agg({'p9' : 'count'}).rename(columns={'p9': 'cause_count'})
#print(df_graphics)
# Vykreslení koláčového grafu pro nehody s následky na životě
df = df[df['p9'] == 'nehoda	s nasledky na zivote'] 

fig, ax = plt.subplots(figsize=(10, 10))
groups = df.groupby('Stav vozovky').agg({"p9": "count"})
groups.plot(kind="pie", ax=ax, startangle=90, title = 'Nehody s následky na životě podle stavu vozovky', subplots=True)
ax.legend(road_labels, loc='upper left')
plt.show()

# Ulozeni grafu
fig.savefig('fig.png')

# Přejmování indexu
groups = groups.rename(index={1: 'povrch suchý', 2: 'povrch suchý - znečištěný', 3: 'povrch mokrý', 4: 'bláto', 5: 'náledí', 6: 'náledí - neposypané', 7: 'olej', 8: 'sněhová vrstva', 9: 'změna stavu vozovky', 0: 'jiný stav povrchu vozovky'})

# Uložení do latexu
groups.to_latex('groups.tex')
print(groups)

# --- Vypisování výsledků ---

# Celkový počet nehod
print('Celkový počet fatálních nehod: ', acc_life)

# Procentuální podíl nehod s následky na životě
acc_fatal = acc_life / sum_acc * 100

print('Procentuální podíl nehod s následky na životě: ', acc_fatal, '%')

# Procentuální podíl nehod vzniklych na suchem povrchu s následky na životě
acc_dry = dry / acc_life * 100

print('Procentuální podíl nehod vzniklych na suchem povrchu s následky na životě: ', acc_dry, '%')

# Procentuální podíl nehod vzniklych na mokrem povrchu s následky na životě
acc_wet = wet / acc_life * 100

print('Procentuální podíl nehod vzniklych na mokrem povrchu s následky na životě: ', acc_wet, '%')

# Procentuální podíl nehod vzniklych na náledí s následky na životě
acc_ice = ice / acc_life * 100

print('Procentuální podíl nehod vzniklych na náledí s následky na životě: ', acc_ice, '%')
