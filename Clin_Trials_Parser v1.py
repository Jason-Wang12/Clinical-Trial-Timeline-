import requests
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import json
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.pyplot as plt

disease = input('please enter disease of interest: ')

url = 'https://clinicaltrials.gov/api/query/study_fields?expr={}&fmt=JSON&max_rnk=1000&fields=NCTId,Condition,BriefTitle,OrgFullName,LeadSponsorClass,PrimaryCompletionDate,PrimaryOutcomeMeasure,InterventionDescription,Phase,InterventionName,InterventionType,DetailedDescription,EnrollmentCount,CentralContactName,CentralContactEMail'.format(disease)
response = requests.get(url)

soup = BeautifulSoup(response.content, 'lxml')
elements = soup.find("p").text
data = json.loads(elements)
data_list = data['StudyFieldsResponse']['StudyFields']
df = pd.DataFrame(data_list)
df = df.drop(columns = 'Rank')
df= df.apply(lambda x: x.str[0])
df.sort_values(by = ['Phase'], inplace = True, ascending = False)
df['Phase'] = df['Phase'].astype(str) #for some reason they're floats, turning to strings
df = df[~df.Phase.str.contains('Phase 4')]  #this is likely repurposing, or other stuff not interesting
df = df[~df.Phase.str.contains('Not Applicable')] #obviously
df = df[~df.Phase.str.contains('nan')]# obviously
df = df[~df.Phase.str.contains('Early Phase 1')]  #too eary to be relevant

df = df[df.InterventionType.isin(['Drug', 'Biological'])] #Only keeps drugs, and biologics in the dataframe, drop all other intervention types

df['ph_num'] = df.Phase.str.extract('(\d+)')#extract numeric of phases
df['ph_num'] = df['ph_num'].astype(float)
df['name_phase'] = [' '.join(i) for i in zip(df['InterventionName'].map(str), df['Phase'])]
#df['name_phase'] = [' '.join(i) for i in zip(df['name_phase'].map(str), df['OrgFullName'])]
print(df.head())

#--Sort the values by the date of completion and what phase they're in
#earliest = input('please enter the earliest date for studies: ')
#latest = input('please enter the latest date for studies: ')

earliest = input('please input the earliest date of interest in format yyyy-mm-dd: ')
latest = input('please input the latest date of interest in format yyyy-mm-dd: ')

df['PrimaryCompletionDate']= pd.to_datetime(df['PrimaryCompletionDate']) #--converts dates to time stamp
dff = df[(df['PrimaryCompletionDate']>earliest)&(df['PrimaryCompletionDate']<latest)]

'''Plotting'''


levels = np.tile([-20, 20, -15, 15,-10, 10, -1, 1],
                 int(np.ceil(len(dff['PrimaryCompletionDate'])/8)))[:len(dff['PrimaryCompletionDate'])]

fig, ax = plt.subplots(figsize=(30, 15), constrained_layout=True)
ax.set(title="{}".format(disease))

markerline, stemline, baseline = ax.stem(dff['PrimaryCompletionDate'], levels,
                                         linefmt="C3-", basefmt="k-",
                                         use_line_collection=True)

plt.setp(markerline, mec="k", mfc="w", zorder=3)


# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(dff['PrimaryCompletionDate'])))

# annotate lines. For each event, we add a text label via annotate, which is 
#offset in units of points from the tip of the event line.
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

#zip will combine two lists into tuples (i.e. a = [2016, 2017], b = [drug a, drug b])
#this becomes [(2016, drug a), (2017, drug b)]

for d, l, r, va in zip(dff['PrimaryCompletionDate'], levels, dff['name_phase'], vert):
    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*10),
                textcoords="offset points", va=va, ha="right", fontsize = 14, rotation = 30)

# format xaxis with 4 month intervals
ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=3))
ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=0, ha="right")

# remove y axis and spines
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)

ax.margins(y=.8)
plt.show()

#dff.to_excel('C:/Users/jwang/Desktop/Python/Biopython/CSV/test.xlsx')

#Statstics on the studies
#--number of trials by phase


phase_dict = dict({'Phase 1':int(len(df[df['Phase'].str.contains('1')])), 
                   'Phase 2': int(len(df[df['Phase'].str.contains('2')])),
                   'Phase 3': int(len(df[df['Phase'].str.contains('3')]))})

phase_df = pd.DataFrame.from_dict(phase_dict, orient = 'index', columns = ['Count'])
phase_df.plot.bar(color = 'rbg', legend = False)
plt.title('Trials by Phase [ALL]')
        

#-- Now let's see trials by phase and by 

phase_df.plot(kind = 'pie', subplots = True)
plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure) #so the legend doesn't overlap with the graph

# -- let's see how much are academia vs industry 
sponsor_counts = df.groupby('LeadSponsorClass').nunique()
sponsor_counts.rename(columns ={'NCTId':'Counts'}, inplace = True)
sponsor_counts_clean = sponsor_counts[['LeadSponsorClass','Counts']]
sponsor_counts_clean=sponsor_counts_clean.drop('LeadSponsorClass', axis = 1)
sponsor_counts_clean= sponsor_counts_clean.T.rename(columns = {'OTHER':'ACADEMIC'}).T
sponsor_counts_clean.plot(kind = 'pie', subplots = True)
plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure) #so the legend doesn't overlap with the graph
plt.ylabel('') #remove the axis label since you don't need it
plt.legend(bbox_to_anchor=(1,0), loc="lower right", bbox_transform=plt.gcf().transFigure) #so the legend doesn't overlap with the graph
plt.show()

# -- let's see which companies are most active in this space over this time period
company_counts = df.groupby('OrgFullName')['NCTId'].nunique().reset_index()
company_counts = company_counts.rename(columns = {'OrgFullName':'Company','NCTId':'Study Counts'})
company_counts = company_counts.sort_values(by = 'Study Counts',ascending = False)
company_counts_2 = company_counts.iloc[:15].copy()
new_row = pd.DataFrame(data = 
                       {'Company':['Other'],
                        'Study Counts': [company_counts['Study Counts'].iloc[15:].sum()]})
company_counts_summarized = pd.concat([company_counts_2, new_row])
company_counts_summarized



earliest2 = input('please input the earliest date of interest in format yyyy-mm-dd: ')
latest2 = input('please input the latest date of interest in format yyyy-mm-dd: ')


df_dl = df[(df['PrimaryCompletionDate']>earliest2)&(df['PrimaryCompletionDate']<latest2)]
#remove large pharma studies
#pharma = ['Bristol-Myers Squibb', 'Boehringer Ingelheim', 'GlaxoSmithKline', 'Pfizer', 'AstraZeneca', 'Novartis']
pharma = ['Vertex Pharmaceuticals Incorporated']
df_dl = df_dl[~df['OrgFullName'].isin(pharma)]


df_dl.to_excel('C:/Users/jason/Google Drive/Python/Biopython/Clinical Trials Parser/Clinical Trial Searches/{} date filtered.xlsx'.format(disease))



#-- try plotting    
#plt.figure(figsize =(,10))
#sns.barplot(x = 'ph_num', y = 'InterventionName', hue = 'Phase', data =df.iloc[:10], palette = 'RdBu_r')

#plot = sns.scatterplot(x = 'PrimaryCompletionDate', y = 'Phase', hue = 'InterventionName', size = 'Phase', data = df.iloc[:10])
#plot.legend(loc='bottom', ncol = 10)

