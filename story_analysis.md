<script>
    var code_show=true; //true -> hide code at first

    function code_toggle() {
        $('div.prompt').hide(); // always hide prompt

        if (code_show){
            $('div.input').hide();
        } else {
            $('div.input').show();
        }
        code_show = !code_show
    }
    $( document ).ready(code_toggle);
</script>

```python
# imports libraries
import pickle										# import/export lists
import math											# mathematical functions
import datetime										# dates
import re 											# regular expression
import pandas as pd									# dataframes
import numpy as np									# numerical computation
import matplotlib.pyplot as plt						# plot graphics
import seaborn as sns								# graphics supplemental
import statsmodels.formula.api as smf				# statistical models
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif)				# vif
```


```python
# opens cleaned data
with open ('df_stories', 'rb') as fp:
    df = pickle.load(fp)
```


```python
# creates subset of data of online stories
df_online = df.loc[df.state == 'online', ].copy()
```


```python
# sets current year
cyear = datetime.datetime.now().year
```

# Fanfiction Story Analysis

In this section, we take a sample of ~5000 stories from fanfiction.net and break down some of their characteristics. Then using all available data from the site, we try to make predictions on the number of reviews a story is expected to have given select features. These predicted values can be used as benchmarks to see whether stories are overperforming or underperforming relative to their peers.

## Data exploration

Let's begin by examining the current state of stories: online, deleted, or missing. Missing stories are stories whose URL has moved due to shifts in the fanfiction achiving system.


```python
# examines state of stories
state = df['state'].value_counts()

# plots chart
(state/np.sum(state)).plot.bar()
plt.xticks(rotation=0)
plt.show()
```


![png](output_9_0.png)


Surprisingly, it appears only about ~60% of stories that were once published still remain on the site! This is in stark contrast to user profiles, where less than 0.1% are deleted.

From this, we can only guess that authors actively take stories down, presumably to hide earlier works as their writing abilities improve or to replace them with rewrites. Authors who delete their profiles and stories that were deleted for fanfiction policy violations would also contribute to these figures.

Now let's examine the volume of stories published across time.


```python
# examines when stories first created
df_online['pub_year'] = [int(row[2]) for row in df_online['published']]
entry = df_online['pub_year'].value_counts().sort_index()

# plots chart
(entry/np.sum(entry)).plot()
plt.xlim([np.min(entry.index.values), cyear-1])
plt.show()
```


![png](output_12_0.png)


We see a large jump starting in the 2010s, peaking around 2013, then a steady decline afterward. Unlike with profiles, you do not see the dips matching the Great Fanfiction Purge of 2002 and 2012.

The decline could be from a variety of factors. One could be competing fanfiction sites.  Most notably, the nonprofit site, Archive of Our Own (AO3), started gaining traction due to its greater inclusivity of works and its tagging system that helps users to filter and search for works.

Another question to ask is if the increasing popularity of fanfiction is fueled by particular fandoms. It is well known in the fanfiction community that fandoms like Star Trek paved the road. Harry Potter and Naruto also held a dominating presence in the 2000s. Later on, we will try to quantify how much each of these fandoms contributed to the volume of fanfiction produced.

### Genres

Now let's look at the distribution across the stories. Note that "General" includes stories that do not have a genre label.


```python
# examines top genres individually
genres_indiv = [item for sublist in df_online['genre'] for item in sublist]
genres_indiv = pd.Series(genres_indiv).value_counts()

# plots chart
(genres_indiv/np.sum(genres_indiv)).plot.bar()
plt.xticks(rotation=90)
plt.show()
```


![png](output_16_0.png)


Romance takes the lead! In fact, ~30% of the genre labels used is "Romance". In second and third place are Humor and Drama respectively.

The least popular genres appear to be Crime, Horror, and Mystery. 

So far, nothing here deviates much from intuition. We'd expect derivative works to focus more on existing character relationships and/or the canonic world, and less on stand-alone plots and twists. 

What about how the genres combine?


```python
# creates contingency table
gen_pairs = df_online.loc[[len(row) > 1 for row in df_online.genre], 'genre']
gen1 = pd.Series([row[0][:3] for row in gen_pairs] + [row[1][:3] for row in gen_pairs])
gen2 = pd.Series([row[1][:3] for row in gen_pairs] + [row[0][:3] for row in gen_pairs])
cross = pd.crosstab(index=gen1, columns=gen2, colnames=[''])
del cross.index.name

# sets border option
pd.options.html.border = 0

# plots table
cm = sns.light_palette('green', as_cmap=True)
cross.style.background_gradient(cmap=cm)
```




<style  type="text/css" >
    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col0 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col1 {
            background-color:  #d8f8d8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col2 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col3 {
            background-color:  #cbf1cb;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col4 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col5 {
            background-color:  #369e36;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col6 {
            background-color:  #a6dca6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col7 {
            background-color:  #92d192;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col8 {
            background-color:  #cbf1cb;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col10 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col11 {
            background-color:  #b6e5b6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col13 {
            background-color:  #69ba69;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col14 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col15 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col16 {
            background-color:  #99d599;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col17 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col18 {
            background-color:  #c0eac0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col0 {
            background-color:  #daf9da;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col1 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col2 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col3 {
            background-color:  #bae7ba;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col4 {
            background-color:  #bde9bd;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col6 {
            background-color:  #d2f4d2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col7 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col8 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col9 {
            background-color:  #9ed79e;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col10 {
            background-color:  #b0e2b0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col11 {
            background-color:  #cef2ce;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col12 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col13 {
            background-color:  #77c277;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col14 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col16 {
            background-color:  #d3f5d3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col17 {
            background-color:  #d6f7d6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col18 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col0 {
            background-color:  #e3fee3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col1 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col3 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col4 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col6 {
            background-color:  #defbde;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col7 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col8 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col10 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col13 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col14 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col16 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col17 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col18 {
            background-color:  #ccf1cc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col0 {
            background-color:  #b7e5b7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col1 {
            background-color:  #90d090;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col2 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col3 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col4 {
            background-color:  #8bcd8b;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col5 {
            background-color:  #bae7ba;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col6 {
            background-color:  #c8efc8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col7 {
            background-color:  #92d192;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col8 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col9 {
            background-color:  #c8efc8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col10 {
            background-color:  #9fd89f;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col12 {
            background-color:  #7dc57d;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col13 {
            background-color:  #088408;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col14 {
            background-color:  #9fd89f;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col15 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col16 {
            background-color:  #acdfac;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col17 {
            background-color:  #a8dda8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col18 {
            background-color:  #269526;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col0 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col1 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col3 {
            background-color:  #d2f4d2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col4 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col5 {
            background-color:  #bae7ba;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col6 {
            background-color:  #bce8bc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col7 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col8 {
            background-color:  #d0f3d0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col9 {
            background-color:  #9cd69c;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col12 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col13 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col14 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col16 {
            background-color:  #d3f5d3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col17 {
            background-color:  #d6f7d6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col18 {
            background-color:  #c0eac0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col0 {
            background-color:  #caf0ca;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col1 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col3 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col4 {
            background-color:  #d3f5d3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col6 {
            background-color:  #daf9da;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col7 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col8 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col13 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col14 {
            background-color:  #9fd89f;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col15 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col16 {
            background-color:  #c9efc9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col17 {
            background-color:  #c7eec7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col18 {
            background-color:  #d9f8d9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col0 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col1 {
            background-color:  #d6f7d6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col2 {
            background-color:  #5cb35c;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col3 {
            background-color:  #daf9da;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col4 {
            background-color:  #99d599;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col5 {
            background-color:  #b0e1b0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col6 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col7 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col8 {
            background-color:  #cef2ce;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col9 {
            background-color:  #94d294;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col10 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col11 {
            background-color:  #daf9da;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col13 {
            background-color:  #8fcf8f;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col14 {
            background-color:  #b0e2b0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col15 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col16 {
            background-color:  #c9efc9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col17 {
            background-color:  #d6f7d6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col18 {
            background-color:  #ccf1cc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col0 {
            background-color:  #dffcdf;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col1 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col3 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col4 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col6 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col7 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col8 {
            background-color:  #e3fee3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col9 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col10 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col12 {
            background-color:  #bce8bc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col13 {
            background-color:  #defbde;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col14 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col15 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col16 {
            background-color:  #acdfac;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col17 {
            background-color:  #a8dda8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col18 {
            background-color:  #ccf1cc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col0 {
            background-color:  #b5e4b5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col1 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col2 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col3 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col4 {
            background-color:  #7ec67e;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col5 {
            background-color:  #bae7ba;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col6 {
            background-color:  #a8dda8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col7 {
            background-color:  #a7dda7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col8 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col10 {
            background-color:  #8dce8d;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col11 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col12 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col13 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col14 {
            background-color:  #8dce8d;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col16 {
            background-color:  #8fcf8f;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col17 {
            background-color:  #d6f7d6;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col18 {
            background-color:  #ccf1cc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col0 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col1 {
            background-color:  #a9dea9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col2 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col3 {
            background-color:  #d9f8d9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col4 {
            background-color:  #56af56;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col5 {
            background-color:  #dbf9db;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col6 {
            background-color:  #8fcf8f;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col7 {
            background-color:  #bce8bc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col8 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col9 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col11 {
            background-color:  #daf9da;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col12 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col13 {
            background-color:  #89cc89;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col14 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col15 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col16 {
            background-color:  #dcfadc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col17 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col18 {
            background-color:  #40a340;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col0 {
            background-color:  #cff3cf;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col1 {
            background-color:  #e0fce0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col2 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col3 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col4 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col6 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col7 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col8 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col9 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col13 {
            background-color:  #dcfadc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col14 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col16 {
            background-color:  #dcfadc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col17 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col18 {
            background-color:  #d9f8d9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col0 {
            background-color:  #dffcdf;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col1 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col3 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col4 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col6 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col7 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col8 {
            background-color:  #d3f5d3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col12 {
            background-color:  #bce8bc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col13 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col14 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col16 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col17 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col18 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col0 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col1 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col3 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col4 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col5 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col6 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col7 {
            background-color:  #bce8bc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col8 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col11 {
            background-color:  #cef2ce;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col13 {
            background-color:  #dffcdf;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col14 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col16 {
            background-color:  #dcfadc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col17 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col18 {
            background-color:  #c0eac0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col0 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col1 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col2 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col3 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col4 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col5 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col6 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col7 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col8 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col9 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col10 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col11 {
            background-color:  #c2ebc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col12 {
            background-color:  #158b15;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col13 {
            background-color:  #e3fee3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col14 {
            background-color:  #58b158;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col15 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col16 {
            background-color:  #008000;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col17 {
            background-color:  #5cb35c;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col18 {
            background-color:  #0d870d;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col0 {
            background-color:  #cff3cf;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col1 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col2 {
            background-color:  #b8e6b8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col3 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col4 {
            background-color:  #dcfadc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col5 {
            background-color:  #bae7ba;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col6 {
            background-color:  #defbde;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col7 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col8 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col9 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col10 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col12 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col13 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col14 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col16 {
            background-color:  #c9efc9;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col17 {
            background-color:  #c7eec7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col18 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col0 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col1 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col3 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col4 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col5 {
            background-color:  #d0f3d0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col6 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col7 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col8 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col10 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col13 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col14 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col16 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col17 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col18 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col0 {
            background-color:  #d8f8d8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col1 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col3 {
            background-color:  #e0fce0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col4 {
            background-color:  #dcfadc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col5 {
            background-color:  #c5edc5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col6 {
            background-color:  #defbde;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col7 {
            background-color:  #68ba68;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col8 {
            background-color:  #ddfbdd;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col9 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col10 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col12 {
            background-color:  #d1f4d1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col13 {
            background-color:  #d2f4d2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col14 {
            background-color:  #b0e2b0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col16 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col17 {
            background-color:  #c7eec7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col18 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col0 {
            background-color:  #cbf1cb;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col1 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col2 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col3 {
            background-color:  #e2fde2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col4 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col5 {
            background-color:  #d0f3d0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col6 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col7 {
            background-color:  #92d192;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col8 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col9 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col10 {
            background-color:  #7bc47b;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col12 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col13 {
            background-color:  #e0fce0;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col14 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col16 {
            background-color:  #d3f5d3;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col17 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col18 {
            background-color:  #99d599;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col0 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col1 {
            background-color:  #c2ecc2;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col2 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col3 {
            background-color:  #d7f7d7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col4 {
            background-color:  #d8f8d8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col5 {
            background-color:  #dbf9db;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col6 {
            background-color:  #e1fde1;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col7 {
            background-color:  #bce8bc;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col8 {
            background-color:  #e4fee4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col9 {
            background-color:  #c8efc8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col10 {
            background-color:  #d4f6d4;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col11 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col12 {
            background-color:  #a7dda7;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col13 {
            background-color:  #d8f8d8;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col14 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col15 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col16 {
            background-color:  #e5ffe5;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col17 {
            background-color:  #8acc8a;
        }    #T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col18 {
            background-color:  #e5ffe5;
        }</style>  
<table id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48" > 
<thead>    <tr> 
        <th class="index_name level0" ></th> 
        <th class="col_heading level0 col0" >Adv</th> 
        <th class="col_heading level0 col1" >Ang</th> 
        <th class="col_heading level0 col2" >Cri</th> 
        <th class="col_heading level0 col3" >Dra</th> 
        <th class="col_heading level0 col4" >Fam</th> 
        <th class="col_heading level0 col5" >Fan</th> 
        <th class="col_heading level0 col6" >Fri</th> 
        <th class="col_heading level0 col7" >Hor</th> 
        <th class="col_heading level0 col8" >Hum</th> 
        <th class="col_heading level0 col9" >Hur</th> 
        <th class="col_heading level0 col10" >Mys</th> 
        <th class="col_heading level0 col11" >Par</th> 
        <th class="col_heading level0 col12" >Poe</th> 
        <th class="col_heading level0 col13" >Rom</th> 
        <th class="col_heading level0 col14" >Sci</th> 
        <th class="col_heading level0 col15" >Spi</th> 
        <th class="col_heading level0 col16" >Sup</th> 
        <th class="col_heading level0 col17" >Sus</th> 
        <th class="col_heading level0 col18" >Tra</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row0" class="row_heading level0 row0" >Adv</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col0" class="data row0 col0" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col1" class="data row0 col1" >7</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col2" class="data row0 col2" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col3" class="data row0 col3" >27</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col4" class="data row0 col4" >10</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col5" class="data row0 col5" >16</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col6" class="data row0 col6" >26</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col7" class="data row0 col7" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col8" class="data row0 col8" >28</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col9" class="data row0 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col10" class="data row0 col10" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col11" class="data row0 col11" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col12" class="data row0 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col13" class="data row0 col13" >132</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col14" class="data row0 col14" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col15" class="data row0 col15" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col16" class="data row0 col16" >8</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col17" class="data row0 col17" >15</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row0_col18" class="data row0 col18" >3</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row1" class="row_heading level0 row1" >Ang</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col0" class="data row1 col0" >7</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col1" class="data row1 col1" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col2" class="data row1 col2" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col3" class="data row1 col3" >44</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col4" class="data row1 col4" >9</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col5" class="data row1 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col6" class="data row1 col6" >8</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col7" class="data row1 col7" >11</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col8" class="data row1 col8" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col9" class="data row1 col9" >31</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col10" class="data row1 col10" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col11" class="data row1 col11" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col12" class="data row1 col12" >11</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col13" class="data row1 col13" >118</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col14" class="data row1 col14" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col15" class="data row1 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col16" class="data row1 col16" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col17" class="data row1 col17" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row1_col18" class="data row1 col18" >18</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row2" class="row_heading level0 row2" >Cri</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col0" class="data row2 col0" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col1" class="data row2 col1" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col2" class="data row2 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col3" class="data row2 col3" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col4" class="data row2 col4" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col5" class="data row2 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col6" class="data row2 col6" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col7" class="data row2 col7" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col8" class="data row2 col8" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col9" class="data row2 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col10" class="data row2 col10" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col11" class="data row2 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col12" class="data row2 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col13" class="data row2 col13" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col14" class="data row2 col14" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col15" class="data row2 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col16" class="data row2 col16" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col17" class="data row2 col17" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row2_col18" class="data row2 col18" >2</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row3" class="row_heading level0 row3" >Dra</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col0" class="data row3 col0" >27</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col1" class="data row3 col1" >44</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col2" class="data row3 col2" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col3" class="data row3 col3" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col4" class="data row3 col4" >20</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col5" class="data row3 col5" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col6" class="data row3 col6" >12</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col7" class="data row3 col7" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col8" class="data row3 col8" >18</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col9" class="data row3 col9" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col10" class="data row3 col10" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col11" class="data row3 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col12" class="data row3 col12" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col13" class="data row3 col13" >232</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col14" class="data row3 col14" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col15" class="data row3 col15" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col16" class="data row3 col16" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col17" class="data row3 col17" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row3_col18" class="data row3 col18" >15</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row4" class="row_heading level0 row4" >Fam</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col0" class="data row4 col0" >10</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col1" class="data row4 col1" >9</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col2" class="data row4 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col3" class="data row4 col3" >20</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col4" class="data row4 col4" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col5" class="data row4 col5" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col6" class="data row4 col6" >17</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col7" class="data row4 col7" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col8" class="data row4 col8" >23</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col9" class="data row4 col9" >32</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col10" class="data row4 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col11" class="data row4 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col12" class="data row4 col12" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col13" class="data row4 col13" >51</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col14" class="data row4 col14" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col15" class="data row4 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col16" class="data row4 col16" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col17" class="data row4 col17" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row4_col18" class="data row4 col18" >3</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row5" class="row_heading level0 row5" >Fan</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col0" class="data row5 col0" >16</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col1" class="data row5 col1" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col2" class="data row5 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col3" class="data row5 col3" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col4" class="data row5 col4" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col5" class="data row5 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col6" class="data row5 col6" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col7" class="data row5 col7" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col8" class="data row5 col8" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col9" class="data row5 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col10" class="data row5 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col11" class="data row5 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col12" class="data row5 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col13" class="data row5 col13" >21</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col14" class="data row5 col14" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col15" class="data row5 col15" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col16" class="data row5 col16" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col17" class="data row5 col17" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row5_col18" class="data row5 col18" >1</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row6" class="row_heading level0 row6" >Fri</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col0" class="data row6 col0" >26</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col1" class="data row6 col1" >8</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col2" class="data row6 col2" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col3" class="data row6 col3" >12</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col4" class="data row6 col4" >17</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col5" class="data row6 col5" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col6" class="data row6 col6" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col7" class="data row6 col7" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col8" class="data row6 col8" >25</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col9" class="data row6 col9" >35</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col10" class="data row6 col10" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col11" class="data row6 col11" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col12" class="data row6 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col13" class="data row6 col13" >93</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col14" class="data row6 col14" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col15" class="data row6 col15" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col16" class="data row6 col16" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col17" class="data row6 col17" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row6_col18" class="data row6 col18" >2</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row7" class="row_heading level0 row7" >Hor</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col0" class="data row7 col0" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col1" class="data row7 col1" >11</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col2" class="data row7 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col3" class="data row7 col3" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col4" class="data row7 col4" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col5" class="data row7 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col6" class="data row7 col6" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col7" class="data row7 col7" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col8" class="data row7 col8" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col9" class="data row7 col9" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col10" class="data row7 col10" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col11" class="data row7 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col12" class="data row7 col12" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col13" class="data row7 col13" >11</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col14" class="data row7 col14" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col15" class="data row7 col15" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col16" class="data row7 col16" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col17" class="data row7 col17" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row7_col18" class="data row7 col18" >2</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row8" class="row_heading level0 row8" >Hum</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col0" class="data row8 col0" >28</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col1" class="data row8 col1" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col2" class="data row8 col2" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col3" class="data row8 col3" >18</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col4" class="data row8 col4" >23</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col5" class="data row8 col5" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col6" class="data row8 col6" >25</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col7" class="data row8 col7" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col8" class="data row8 col8" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col9" class="data row8 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col10" class="data row8 col10" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col11" class="data row8 col11" >19</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col12" class="data row8 col12" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col13" class="data row8 col13" >241</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col14" class="data row8 col14" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col15" class="data row8 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col16" class="data row8 col16" >9</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col17" class="data row8 col17" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row8_col18" class="data row8 col18" >2</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row9" class="row_heading level0 row9" >Hur</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col0" class="data row9 col0" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col1" class="data row9 col1" >31</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col2" class="data row9 col2" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col3" class="data row9 col3" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col4" class="data row9 col4" >32</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col5" class="data row9 col5" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col6" class="data row9 col6" >35</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col7" class="data row9 col7" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col8" class="data row9 col8" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col9" class="data row9 col9" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col10" class="data row9 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col11" class="data row9 col11" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col12" class="data row9 col12" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col13" class="data row9 col13" >99</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col14" class="data row9 col14" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col15" class="data row9 col15" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col16" class="data row9 col16" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col17" class="data row9 col17" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row9_col18" class="data row9 col18" >13</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row10" class="row_heading level0 row10" >Mys</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col0" class="data row10 col0" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col1" class="data row10 col1" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col2" class="data row10 col2" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col3" class="data row10 col3" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col4" class="data row10 col4" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col5" class="data row10 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col6" class="data row10 col6" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col7" class="data row10 col7" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col8" class="data row10 col8" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col9" class="data row10 col9" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col10" class="data row10 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col11" class="data row10 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col12" class="data row10 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col13" class="data row10 col13" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col14" class="data row10 col14" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col15" class="data row10 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col16" class="data row10 col16" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col17" class="data row10 col17" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row10_col18" class="data row10 col18" >1</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row11" class="row_heading level0 row11" >Par</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col0" class="data row11 col0" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col1" class="data row11 col1" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col2" class="data row11 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col3" class="data row11 col3" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col4" class="data row11 col4" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col5" class="data row11 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col6" class="data row11 col6" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col7" class="data row11 col7" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col8" class="data row11 col8" >19</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col9" class="data row11 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col10" class="data row11 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col11" class="data row11 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col12" class="data row11 col12" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col13" class="data row11 col13" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col14" class="data row11 col14" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col15" class="data row11 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col16" class="data row11 col16" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col17" class="data row11 col17" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row11_col18" class="data row11 col18" >0</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row12" class="row_heading level0 row12" >Poe</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col0" class="data row12 col0" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col1" class="data row12 col1" >11</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col2" class="data row12 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col3" class="data row12 col3" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col4" class="data row12 col4" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col5" class="data row12 col5" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col6" class="data row12 col6" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col7" class="data row12 col7" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col8" class="data row12 col8" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col9" class="data row12 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col10" class="data row12 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col11" class="data row12 col11" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col12" class="data row12 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col13" class="data row12 col13" >10</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col14" class="data row12 col14" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col15" class="data row12 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col16" class="data row12 col16" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col17" class="data row12 col17" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row12_col18" class="data row12 col18" >3</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row13" class="row_heading level0 row13" >Rom</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col0" class="data row13 col0" >132</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col1" class="data row13 col1" >118</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col2" class="data row13 col2" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col3" class="data row13 col3" >232</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col4" class="data row13 col4" >51</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col5" class="data row13 col5" >21</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col6" class="data row13 col6" >93</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col7" class="data row13 col7" >11</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col8" class="data row13 col8" >241</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col9" class="data row13 col9" >99</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col10" class="data row13 col10" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col11" class="data row13 col11" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col12" class="data row13 col12" >10</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col13" class="data row13 col13" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col14" class="data row13 col14" >8</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col15" class="data row13 col15" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col16" class="data row13 col16" >24</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col17" class="data row13 col17" >9</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row13_col18" class="data row13 col18" >17</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row14" class="row_heading level0 row14" >Sci</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col0" class="data row14 col0" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col1" class="data row14 col1" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col2" class="data row14 col2" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col3" class="data row14 col3" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col4" class="data row14 col4" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col5" class="data row14 col5" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col6" class="data row14 col6" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col7" class="data row14 col7" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col8" class="data row14 col8" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col9" class="data row14 col9" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col10" class="data row14 col10" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col11" class="data row14 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col12" class="data row14 col12" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col13" class="data row14 col13" >8</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col14" class="data row14 col14" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col15" class="data row14 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col16" class="data row14 col16" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col17" class="data row14 col17" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row14_col18" class="data row14 col18" >0</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row15" class="row_heading level0 row15" >Spi</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col0" class="data row15 col0" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col1" class="data row15 col1" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col2" class="data row15 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col3" class="data row15 col3" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col4" class="data row15 col4" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col5" class="data row15 col5" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col6" class="data row15 col6" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col7" class="data row15 col7" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col8" class="data row15 col8" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col9" class="data row15 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col10" class="data row15 col10" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col11" class="data row15 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col12" class="data row15 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col13" class="data row15 col13" >5</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col14" class="data row15 col14" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col15" class="data row15 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col16" class="data row15 col16" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col17" class="data row15 col17" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row15_col18" class="data row15 col18" >0</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row16" class="row_heading level0 row16" >Sup</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col0" class="data row16 col0" >8</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col1" class="data row16 col1" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col2" class="data row16 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col3" class="data row16 col3" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col4" class="data row16 col4" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col5" class="data row16 col5" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col6" class="data row16 col6" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col7" class="data row16 col7" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col8" class="data row16 col8" >9</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col9" class="data row16 col9" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col10" class="data row16 col10" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col11" class="data row16 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col12" class="data row16 col12" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col13" class="data row16 col13" >24</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col14" class="data row16 col14" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col15" class="data row16 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col16" class="data row16 col16" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col17" class="data row16 col17" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row16_col18" class="data row16 col18" >0</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row17" class="row_heading level0 row17" >Sus</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col0" class="data row17 col0" >15</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col1" class="data row17 col1" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col2" class="data row17 col2" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col3" class="data row17 col3" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col4" class="data row17 col4" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col5" class="data row17 col5" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col6" class="data row17 col6" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col7" class="data row17 col7" >4</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col8" class="data row17 col8" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col9" class="data row17 col9" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col10" class="data row17 col10" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col11" class="data row17 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col12" class="data row17 col12" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col13" class="data row17 col13" >9</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col14" class="data row17 col14" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col15" class="data row17 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col16" class="data row17 col16" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col17" class="data row17 col17" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row17_col18" class="data row17 col18" >6</td> 
    </tr>    <tr> 
        <th id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48level0_row18" class="row_heading level0 row18" >Tra</th> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col0" class="data row18 col0" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col1" class="data row18 col1" >18</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col2" class="data row18 col2" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col3" class="data row18 col3" >15</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col4" class="data row18 col4" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col5" class="data row18 col5" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col6" class="data row18 col6" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col7" class="data row18 col7" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col8" class="data row18 col8" >2</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col9" class="data row18 col9" >13</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col10" class="data row18 col10" >1</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col11" class="data row18 col11" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col12" class="data row18 col12" >3</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col13" class="data row18 col13" >17</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col14" class="data row18 col14" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col15" class="data row18 col15" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col16" class="data row18 col16" >0</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col17" class="data row18 col17" >6</td> 
        <td id="T_bac7e3cc_7c80_11e7_8e90_e82aeaf71f48row18_col18" class="data row18 col18" >0</td> 
    </tr></tbody> 
</table> 



In terms of how genres cross, romance appears to pair with almost everything. Romance is particularly common with drama (the romantic drama) and humor (the rom-com). The only genre that shies away from romance is parody, which goes in hand with humor instead. 

The second most crossed genre is adventure, which is often combined with fantasy, sci-fi, mystery, or suspense. 

The third genre to note is angst, which is often combined with horror, poetry, or tragedy.

### Ratings

The breakdown of how stories are rated are given below.


```python
# examines state of stories
rated = df_online['rated'].value_counts()
rated.plot.pie(autopct='%.f', figsize=(5,5))

plt.show()
```


![png](output_23_0.png)


~40% of stories are rated T, ~40% rated K or K+, and ~20% are rated M.

### Media and fandoms

As for 2017, fanfiction.net has nine different media categories, plus crossovers. The breakdown of these is given below:


```python
# examines distribution of media
media = df_online['media'].value_counts()
(media/np.sum(media)).plot.bar()
plt.xticks(rotation=90)

plt.show()
```


![png](output_27_0.png)


Anime/Manga is the most popular media, taking up approximately ~30% of all works. TV Shows and Books both contribute to ~20% each.

What about by fandom?


```python
# examines distribution of media
fandom = df_online['fandom'].value_counts()
(fandom[:10]/np.sum(fandom)).plot.bar()
plt.xticks(rotation=90)

plt.show()
```


![png](output_30_0.png)


The most popular fandom is, unsurprisingly, Harry Potter. However, it still consitutes a much smaller portion of the fanfiction base than initially assumed, at only ~10%.

One question we asked earlier is what fandoms contributed to the increases in stories over time.


```python
df_online['top_fandom'] = df_online['fandom']
nottop = [row not in fandom[:10].index.values for row in df_online['fandom']]
df_online.loc[nottop, 'top_fandom'] = 'Other'
```


```python
entry_fandom = pd.crosstab(df_online.pub_year, df_online.top_fandom)
entry_fandom = entry_fandom[np.append(fandom[:5].index.values, ['Other'])][:-1]

# plots chart
(entry_fandom/np.sum(entry)).plot.bar(stacked=True)
plt.axes().get_xaxis().set_label_text('')
plt.legend(title=None, frameon=False)
plt.show()
```


![png](output_34_0.png)


It would appear that backin the year 2000, Harry Potter constituted nearly half of the fanfictions published. However, the overall growth in fanfiction is due to many other fandoms jumping in, with no one particular fandom holding sway.

Of the top 5 fandoms, Harry Potter and Naruto prove to be the most persistent in holding their volumes per year. Twilight saw a giant spike in popularity in 2009 and 2010 but faded since.

### Word count, chapter length and completion status

Let's take a look at the distribution of word and chapter lengths.


```python
# examines distribution of number of words
df_online['words1k'] = df_online['words']/1000

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
sns.kdeplot(df_online['words1k'], shade=True, bw=.5, legend=False, ax=ax1)
sns.kdeplot(df_online['words1k'], shade=True, bw=.5, legend=False, ax=ax2)
plt.xlim(0,100)

plt.show()
```


![png](output_38_0.png)


The bulk of stories appear to be less than 50 thousand words, with a high proportion between 0-20 thousand words. In other words, we have a significant proportion of short stories and novelettes, and some novellas. Novels become more rare. Finally, there are a few "epics", ranging from 200 thousand to 600 thousand words.

The number of chapters per story, unsurprisingly, follows a similarly skewed distribution.


```python
# examines distribution of number of chapters
df_online['chapters'] = df_online['chapters'].fillna(1)
df_online['chapters'].plot.hist(normed=True, 
                                bins=np.arange(1, max(df_online.chapters)+1, 1))

plt.show()
```


![png](output_41_0.png)


Stories with over 20 chapters become exceedingly rare.

How often are stories completed?


```python
# examines distribution of story status
status = df_online['status'].value_counts()
status.plot.pie(autopct='%.f', figsize=(5,5))

plt.show()
```


![png](output_44_0.png)


This is unexpected. It looks to be about an even split between completed and incompleted stories.

Let's see what types of stories are the completed ones.


```python
complete = df_online.loc[df_online.status == 'Complete', 'chapters']
incomplete = df_online.loc[df_online.status == 'Incomplete', 'chapters']

plt.hist([complete, incomplete], normed=True, range=[1,10])

plt.show()
```


![png](output_47_0.png)


Oneshots explain the large proportion of completed stories.

### Publication timing

Do authors publish more frequently on certain months or days?


```python
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months = ['NA', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 
          'August', 'Septemeber', 'October', 'November', 'December']
```


```python
# examines when stories first created
df_online['pub_month'] = [months[int(row[0])] for row in df_online['published']]
month = df_online['pub_month'].value_counts()
(month/np.sum(month)).plot.bar()
plt.xticks(rotation=90)
plt.axhline(y=0.0833, color='red')

plt.show()
```


![png](output_52_0.png)


It appears some months are more popular than others. September, October, and November are the least popular months. Given that the majority of the user base is from the United States, and presumably children and young adults, this is perhaps due to the timing of the academic calendar -- school begins in the fall. 

Similarly, the three most popular months (December, July, and April) coincides with winter vacation, summer vacation, and spring break respectively.


```python
# examines when stories first created
dayofweek = [days[datetime.date(int(row[2]), int(row[0]), int(row[1])).weekday()] 
             for row in df_online['published']]
dayofweek = pd.Series(dayofweek).value_counts()
(dayofweek/np.sum(dayofweek)).plot.bar()
plt.xticks(rotation=90)
plt.axhline(y=0.143, color='red')

plt.show()
```


![png](output_54_0.png)


As for days of the week, publications are least likely to happen on a Friday.

## Regression Analysis

In this section, we will try to predict the number of stories an (active) user would write based off the number of years have they been on the site, the number of authors/stories they have favorited, and whether or not they are in a community.


```python
df_online.columns.values
```




    array(['storyid', 'userid', 'title', 'summary', 'media', 'fandom', 'rated',
           'language', 'genre', 'characters', 'chapters', 'words', 'reviews',
           'favs', 'follows', 'updated', 'published', 'status', 'state',
           'pub_year'], dtype=object)




```python
df_online['ratedM'] = [row == 'M' for row in df_online['rated']]
df_online['age'] = [cyear - int(row) for row in df_online['pub_year']]
df_online['fansize'] = [fandom[row] for row in df_online['fandom']]
df_online['complete'] = [row == 'Complete' for row in df_online['status']]
```


```python
# runs OLS regression
formula = 'reviews ~ chapters + words1k + ratedM + age + fansize + complete'
reg = smf.ols(data=df_online, formula=formula).fit()
print(reg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                reviews   R-squared:                       0.190
    Model:                            OLS   Adj. R-squared:                  0.188
    Method:                 Least Squares   F-statistic:                     103.0
    Date:                Tue, 08 Aug 2017   Prob (F-statistic):          8.65e-117
    Time:                        10:14:15   Log-Likelihood:                -15926.
    No. Observations:                2639   AIC:                         3.187e+04
    Df Residuals:                    2632   BIC:                         3.191e+04
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept           -6.6472      4.920     -1.351      0.177     -16.295       3.001
    ratedM[T.True]      19.9886      5.055      3.954      0.000      10.076      29.901
    complete[T.True]     3.7254      3.996      0.932      0.351      -4.111      11.562
    chapters             3.2265      0.399      8.090      0.000       2.444       4.009
    words1k              1.1135      0.099     11.298      0.000       0.920       1.307
    age                  0.2418      0.495      0.488      0.626      -0.730       1.213
    fansize              0.0387      0.022      1.794      0.073      -0.004       0.081
    ==============================================================================
    Omnibus:                     6223.232   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         53684044.481
    Skew:                          23.087   Prob(JB):                         0.00
    Kurtosis:                     700.201   Cond. No.                         322.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# runs OLS regression
formula = 'lnreviews ~ lnchapters + lnwords1k + ratedM + age + fansize + complete'
reg = smf.ols(data=df_online, formula=formula).fit()
print(reg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:              lnreviews   R-squared:                       0.434
    Model:                            OLS   Adj. R-squared:                  0.433
    Method:                 Least Squares   F-statistic:                     336.5
    Date:                Tue, 08 Aug 2017   Prob (F-statistic):          6.35e-321
    Time:                        10:14:25   Log-Likelihood:                -3435.8
    No. Observations:                2639   AIC:                             6886.
    Df Residuals:                    2632   BIC:                             6927.
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    Intercept            1.1808      0.046     25.717      0.000       1.091       1.271
    ratedM[T.True]       0.1468      0.045      3.262      0.001       0.059       0.235
    complete[T.True]     0.2795      0.037      7.656      0.000       0.208       0.351
    lnchapters           0.4958      0.028     17.563      0.000       0.440       0.551
    lnwords1k            0.2247      0.019     11.710      0.000       0.187       0.262
    age                  0.0442      0.004     10.127      0.000       0.036       0.053
    fansize              0.0006      0.000      2.925      0.003       0.000       0.001
    ==============================================================================
    Omnibus:                      113.198   Durbin-Watson:                   1.996
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              203.572
    Skew:                           0.334   Prob(JB):                     6.24e-45
    Kurtosis:                       4.185   Cond. No.                         337.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# runs OLS regression
formula = 'lnfavs ~ lnchapters + lnwords1k + ratedM + age + fansize'
reg = smf.ols(data=df_online, formula=formula).fit()
print(reg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 lnfavs   R-squared:                       0.216
    Model:                            OLS   Adj. R-squared:                  0.215
    Method:                 Least Squares   F-statistic:                     139.2
    Date:                Tue, 08 Aug 2017   Prob (F-statistic):          1.14e-130
    Time:                        10:10:56   Log-Likelihood:                -3825.4
    No. Observations:                2527   AIC:                             7663.
    Df Residuals:                    2521   BIC:                             7698.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    Intercept          2.0682      0.047     43.806      0.000       1.976       2.161
    ratedM[T.True]     0.3411      0.056      6.080      0.000       0.231       0.451
    lnchapters        -0.0870      0.035     -2.486      0.013      -0.156      -0.018
    lnwords1k          0.3969      0.025     15.956      0.000       0.348       0.446
    age               -0.0308      0.006     -5.409      0.000      -0.042      -0.020
    fansize            0.0010      0.000      4.207      0.000       0.001       0.001
    ==============================================================================
    Omnibus:                       92.491   Durbin-Watson:                   1.943
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              102.129
    Skew:                           0.475   Prob(JB):                     6.65e-23
    Kurtosis:                       3.257   Cond. No.                         286.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# creates copy of only active users
df_active = df_profile.loc[df_profile.status != 'inactive', ].copy()

# creates age variable
df_active['age'] = 17 - pd.to_numeric(df_active['join_year'])
df_active.loc[df_active.age < 0, 'age'] = df_active.loc[df_active.age < 0, 'age'] + 100
df_active = df_active[['st', 'fa', 'fs', 'cc', 'age']]

# turns cc into binary
df_active.loc[df_active['cc'] > 0, 'cc'] = 1
```

## Multicollinearity


```python
# displays correlation matrix
df_active.corr()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>st</th>
      <th>fa</th>
      <th>fs</th>
      <th>cc</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>st</th>
      <td>1.000000</td>
      <td>0.089321</td>
      <td>0.142494</td>
      <td>0.052937</td>
      <td>0.170821</td>
    </tr>
    <tr>
      <th>fa</th>
      <td>0.089321</td>
      <td>1.000000</td>
      <td>0.706184</td>
      <td>0.017645</td>
      <td>0.007866</td>
    </tr>
    <tr>
      <th>fs</th>
      <td>0.142494</td>
      <td>0.706184</td>
      <td>1.000000</td>
      <td>0.118110</td>
      <td>0.011833</td>
    </tr>
    <tr>
      <th>cc</th>
      <td>0.052937</td>
      <td>0.017645</td>
      <td>0.118110</td>
      <td>1.000000</td>
      <td>0.113621</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.170821</td>
      <td>0.007866</td>
      <td>0.011833</td>
      <td>0.113621</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# creates design_matrix 
X = df_active
X['intercept'] = 1

# displays variance inflation factor
vif_results = pd.DataFrame()
vif_results['VIF Factor'] = [vif(X.values, i) for i in range(X.shape[1])]
vif_results['features'] = X.columns
vif_results
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.051990</td>
      <td>st</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.013037</td>
      <td>fa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.064973</td>
      <td>fs</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.036716</td>
      <td>cc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.042636</td>
      <td>age</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.824849</td>
      <td>intercept</td>
    </tr>
  </tbody>
</table>
</div>



Results indicate there is some correlation between two of the independent variables: 'fa' and 'fs', implying one of them may not be necessary in the model.

## Nonlinearity

We know from earlier distributions that some of the variables are heavily right-skewed. We created some scatter plots to confirm that the assumption of linearity holds.


```python
df_online['lnreviews'] = np.log(df_online['reviews']+1)
df_online['lnchapters'] = np.log(df_online['chapters'])
df_online['lnwords1k'] = np.log(df_online['words1k'])

sns.pairplot(data=df_online, y_vars=['lnreviews'], x_vars=['lnchapters', 'lnwords1k', 'age'])
sns.pairplot(data=df_online, y_vars=['reviews'], x_vars=['chapters', 'words', 'age'])

plt.show()
```


![png](output_70_0.png)



![png](output_70_1.png)



```python
df_online['lnfavs'] = np.log(df_online['favs']+1)

sns.pairplot(data=df_online, y_vars=['lnfavs'], x_vars=['lnchapters', 'lnwords1k', 'age'])
sns.pairplot(data=df_online, y_vars=['favs'], x_vars=['chapters', 'words', 'age'])

plt.show()
```


![png](output_71_0.png)



![png](output_71_1.png)


The data is clustered around the zeros. Let's try a log transformation.

## Regression Model


```python
# runs OLS regression
formula = 'st ~ fa + fs + cc + age'
reg = smf.ols(data=df_active, formula=formula).fit()
print(reg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     st   R-squared:                       0.199
    Model:                            OLS   Adj. R-squared:                  0.196
    Method:                 Least Squares   F-statistic:                     61.31
    Date:                Thu, 03 Aug 2017   Prob (F-statistic):           2.70e-46
    Time:                        18:53:22   Log-Likelihood:                -757.62
    No. Observations:                 992   AIC:                             1525.
    Df Residuals:                     987   BIC:                             1550.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0338      0.029     -1.171      0.242      -0.090       0.023
    fa             0.1482      0.029      5.150      0.000       0.092       0.205
    fs             0.0401      0.018      2.287      0.022       0.006       0.075
    cc             0.6732      0.148      4.538      0.000       0.382       0.964
    age            0.0290      0.004      6.847      0.000       0.021       0.037
    ==============================================================================
    Omnibus:                      583.226   Durbin-Watson:                   2.123
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5544.765
    Skew:                           2.580   Prob(JB):                         0.00
    Kurtosis:                      13.370   Cond. No.                         60.5
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The log transformations helped increase the fit from and R-squared of ~0.05 to ~0.20.

From these results, we can see that:

* A 1% change in number of authors favorited is associated with a ~15% change in the number of stories written.
* A 1% change in number of stories favorited is associated with a ~4% change in the number of stories written.
* Being in a community is associated with a ~0.7 increase in the number of stories written.
* One more year on the site is associated with a ~3% change in the number of stories written.

We noted earlier that 'fa' and 'fs' had a correlation of ~0.7. As such, we reran the regression without 'fa' first, then again without 'fs'. The model without 'fs' yielded a better fit (R-squared), as well as AIC and BIC.


```python
# runs OLS regression
formula = 'st ~ fa + cc + age'
reg = smf.ols(data=df_active, formula=formula).fit()
print(reg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     st   R-squared:                       0.195
    Model:                            OLS   Adj. R-squared:                  0.192
    Method:                 Least Squares   F-statistic:                     79.67
    Date:                Thu, 03 Aug 2017   Prob (F-statistic):           3.69e-46
    Time:                        18:53:27   Log-Likelihood:                -760.24
    No. Observations:                 992   AIC:                             1528.
    Df Residuals:                     988   BIC:                             1548.
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.0169      0.028     -0.605      0.545      -0.072       0.038
    fa             0.1989      0.018     10.843      0.000       0.163       0.235
    cc             0.7102      0.148      4.806      0.000       0.420       1.000
    age            0.0281      0.004      6.636      0.000       0.020       0.036
    ==============================================================================
    Omnibus:                      592.647   Durbin-Watson:                   2.130
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5757.058
    Skew:                           2.627   Prob(JB):                         0.00
    Kurtosis:                      13.568   Cond. No.                         59.7
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


Without 'fs', we lost some information but not much:

* A 1% change in number of authors favorited is associated with a ~20% change in the number of stories written.
* Being in a community is associated with a ~0.7 increase in the number of stories written.
* One more year on the site is associated with a ~3% change in the number of stories written.

All these results seem to confirm a basic intuition that the more active an user reads (as measured by favoriting authors and stories), the likely it is that user will write more stories. Being longer on the site and being part of a community is also correlated to publications.

To get a sense of the actual magnitude of these effects, let's attempt some plots:


```python
def graph(formula, x_range):  
    y = np.array(x_range)
    x = formula(y)
    plt.plot(y,x)  

graph(lambda x : (np.exp(reg.params[0]+reg.params[1]*(np.log(x-1)))), 
      range(2,100,1))
graph(lambda x : (np.exp(reg.params[0]+reg.params[1]*(np.log(x-1))+reg.params[2])), 
      range(2,100,1))

plt.show() 
```


![png](output_79_0.png)



```python
ages = [0, 1, 5, 10, 15]
for age in ages:
    graph(lambda x : (np.exp(reg.params[0]+reg.params[1]*(np.log(x-1))+reg.params[3]*age)), 
          range(2,100,1))

plt.show() 
```


![png](output_80_0.png)

