# MedPhysStatsCourse
Materials for the MedPhys Stats course

This consists of the example dataset (`example.csv`), and a Python module containing a few convenience functions to implement some of the functionality missing from Python (`stats_course.py`. 

## Importing the Python module from Google Colab:

The easiest way to import this module is by placing this code snippet at the head of your notebook. Note that it will also import all the modules necessary for the course.
```
import urllib
url = 'https://raw.githubusercontent.com/jdtournier/MedPhysStatsCourse/main/stats_course.py'

with open ('stats_course.py', 'wb') as data:
  data.write (urllib.request.urlopen(url).read())

from stats_course import *
```

## Content of Python module:

- `load_data(url)`

    load the data at the specified URL into a pandas dataframe

- `IQR(var)`

    Compute the interquartile range for the variable `var`
    
- `confint(data, level=0.95)`
    
    Return confidence interval for the mean at the relevant level.
    
    By default, the 95% level is assumed. This is computed using t-values, 
    and is therefore suitable for small samples.

- `add_Normal(axes, data, width, color='orange')`
    
    Add the Normal curve on top of e.g. a histogram.
    
    This requires a handle to the axes on which to add the curve (as 
    returned by e.g. the corresponding seaborn histplot() call), 
    the data for the relevant variable, and the bin width used in the 
    original histogram. 
    
    For example:
    
        ax = sns.histplot (data=df['phadm'], binrange=[6.925, 7.525], binwidth=0.05);
        add_Normal (ax, df['phadm'], 0.05)

- `VIF(model)`
    
    Compute the variance inflation factors for the variables in `model`.

- `stepwise_forward_R2(dataframe, dependent, candidates, selected=[])`
    
    Perform stepwise forward linear regression by maximising R²

- `stepwise_backward_R2(dataframe, dependent, candidates)`
    
    Perform stepwise backward linear regression by maximising R²

- `stepwise_forward_pval(dataframe, dependent, candidates, pvalue_threshold=0.05, selected=[])`
    
    Perform stepwise forward linear regression by inspecting p-values

- `stepwise_backward_pval(dataframe, dependent, candidates, pvalue_threshold=0.1)`
    
    Perform stepwise backward linear regression by inspecting p-values
