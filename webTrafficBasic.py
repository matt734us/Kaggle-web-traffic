
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd

print('Reading data...')
key_1 = pd.read_csv('./Documents/workspace/webTrafFor/key_1.csv')
train_1 = pd.read_csv('./Documents/workspace/webTrafFor/train_1.csv')
ss_1 = pd.read_csv('./Documents/workspace/webTrafFor/sample_submission_1.csv')

print('Processing...')
#create unique ids list from id column in key_1 dataframe
ids = key_1.Id.values
#create unique pages list from Page column in key_1 dataframe
pages = key_1.Page.values


# In[ ]:

print('key_1...')
#creates d_pages dictionary
d_pages = {}
#iterates through zipped ids and pages
for id, page in zip(ids, pages):
#returns name of page minus the date
    d_pages[id] = page[:-11]

#List of Pages
pages = train_1.Page.values

#Using the median number of visits over the past 56 days to make a prediction
visits = np.nan_to_num(np.round(np.nanmedian(train_1.drop('Page', axis=1).values[:, -56:], axis=1)))


# In[ ]:

#Create dictionary of pages with median number of visits for last 56 days
d_visits = {}
for page, visits_number in zip(pages, visits):
    d_visits[page] = visits_number

#Modify sample submission with the predicted number of visits
print('Modifying sample submission...')
ss_ids = ss_1.Id.values
ss_visits = ss_1.Visits.values

for i, ss_id in enumerate(ss_ids):
    ss_visits[i] = d_visits[d_pages[ss_id]]

print('Saving submission...')
#Create dataframe for ids and number of predicted visits from sample submission file
subm = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})
#Create submission CSV file
subm.to_csv('./Documents/workspace/webTrafFor/submission.csv', index=False)


# In[ ]:



