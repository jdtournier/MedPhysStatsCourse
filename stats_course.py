import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
from google.colab import drive

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,5)

def load_data (url):
  import urllib
  with open ('example.csv', 'wb') as data:  
    data.write (urllib.request.urlopen(url).read())
  return pd.read_csv ('example.csv', na_values=['', ' '])
  

def IQR (var):
  return var.quantile(0.75) - var.quantile(0.25)
  

def add_Normal (axes, data, width, color='orange'):
  x = np.linspace (axes.get_xlim()[0], axes.get_xlim()[1])
  axes.plot (x, width*len(data)*stats.norm.pdf(x, data.mean(), data.std()), color=color);
  

def confint (data, level=0.95):
  z = stats.norm.ppf(1-(1-level)/2)
  m = data.mean()
  s = data.sem()
  return [m-z*s, m+z*s]
  
  
def stepwise_forward_R2 (dataframe, dependent, candidates, selected=[]):
  """Perform stepwise forward linear regression by maximising R²"""

  # helper function to keep following code clean:
  def report (model):
    print ("  {} : R² = {}".format (model.model.formula, model.rsquared_adj))

  # initial formula, with a single constant as independent 
  # if 'selected' is empty, or the contents of 'selected' otherwise.
  formula = "{} ~ {}".format (dependent, ' + '.join (selected) if len(selected) else '1')
  best_model = smf.ols (formula, data=df).fit()
  report (best_model)

  # blank line
  print ()

  # iterate while there are still independent candidates to check:
  while len (candidates) > 0:
    # run model for all candidates:
    models = []
    for var in candidates:
      formula = "{} ~ {}".format (dependent, ' + '.join (selected + [ var ]))
      models.append (smf.ols (formula, data=df).fit())
      report (models[-1])

    # identify best model and stop if not better than current R²:
    idx = np.argmax ([ model.rsquared_adj for model in models ])
    if models[idx].rsquared_adj <= best_model.rsquared_adj:
      break

    # update model and iterate:
    best_model = models[idx]
    selected.append (candidates.pop(idx))
    print ("\nselected {} with R² = {}\n".format (best_model.model.formula, best_model.rsquared_adj))
    
  print ("\nfinal model:")
  report (best_model)
  return best_model
  
  
def stepwise_backward_R2 (dataframe, dependent, candidates):
  """Perform stepwise backward linear regression by maximising R²"""

  # helper function to keep following code clean:
  def report (model):
    print ("  {} : R² = {}".format (model.model.formula, model.rsquared_adj))

  # initial formula, with all candidates as independents:
  formula = "{} ~ {}".format (dependent, ' + '.join (candidates))   # <== (1)
  best_model = smf.ols (formula, data=df).fit()
  report (best_model)

  # blank line
  print ()

  # iterate while there are still independent candidates to check:
  while len (candidates) > 0:
    # run model for all candidates:
    models = []
    for var in candidates:
      formula = "{} ~ {}".format (dependent, ' + '.join ([ x for x in candidates if x != var ]))    # <== (2)
      models.append (smf.ols (formula, data=df).fit())
      report (models[-1])

    # identify best model and stop if not better than current R²:
    idx = np.argmax ([ model.rsquared_adj for model in models ])
    if models[idx].rsquared_adj <= best_model.rsquared_adj:
      break

    # update model and iterate:
    best_model = models[idx]
    candidates.pop(idx)              # <== (3)
    print ("\nselected {} with R² = {}\n".format (best_model.model.formula, best_model.rsquared_adj))
    
  print ("\nfinal model:")
  report (best_model)
  return best_model
  


def stepwise_forward_pval (dataframe, dependent, candidates, pvalue_threshold=0.05, selected=[]):
  """Perform stepwise forward linear regression by inspecting p-values"""

  # helper function to keep following code clean:
  def report (model):
    print ("  {} : R² = {}, pvalue = {}".format (
        model.model.formula, model.rsquared_adj, model.pvalues[-1]))
    
  # initial formula, with a single constant as independent 
  # if 'selected' is empty, or the contents of 'selected' otherwise.
  formula = "{} ~ {}".format (dependent, ' + '.join (selected) if len(selected) else '1')
  best_model = smf.ols (formula, data=df).fit()
  report (best_model)

  # blank line
  print ()

  # iterate while there are still independent candidates to check:
  while len (candidates) > 0:
    # run model for all candidates:
    models = []
    for var in candidates:
      formula = "{} ~ {}".format (dependent, ' + '.join (selected + [ var ]))
      models.append (smf.ols (formula, data=df).fit())
      report (models[-1])

    # identify best model and stop if not below threshold:
    idx = np.argmin ([ m.pvalues[-1] for m in models])
    if models[idx].pvalues[-1] > pvalue_threshold:
      break

    # update model and iterate:
    best_model = models[idx]
    selected.append (candidates.pop (idx))
    print ("\nselected variable {}:".format (selected[-1]))
    report (best_model)
    print()
    
  print ("\nfinal model:")
  report (best_model)
  return best_model
  

def stepwise_backward_pval (dataframe, dependent, candidates, pvalue_threshold=0.1):
  """Perform stepwise backward linear regression by inspecting p-values"""

  # helper function to keep following code clean:
  def report (model):
    print ("  {} : R² = {}".format (model.model.formula, model.rsquared_adj))
    for p in model.pvalues.items():
      print ("    {}, p = {}".format (p[0], p[1]))

  # iterate while there are still independent candidates to check:
  while len (candidates) > 0:
    formula = "{} ~ {}".format (dependent, ' + '.join (candidates))
    model = smf.ols (formula, data=df).fit()
    report (model)

    # find variable with highest p-values (ignoring intercept):
    pvalues = model.pvalues[1:]
    idx = np.argmax (pvalues)
    # if worst variable is below threshold, stop:
    if pvalues[idx] <= pvalue_threshold:
      break

    # update model and iterate:
    print ("\nremoving {} with p = {}\n".format (candidates[idx], pvalues[idx]))
    candidates.pop(idx)
    
  print ("\nfinal model:")
  report (model)
  return model
