import sys
import pandas as pd

# Read the text file and split the data into interactions
with open(sys.argv[1], 'r') as file:
    data = file.read().split('\nid=')

# Initialize lists to store queries and answers
queries = []
answers = []

# Process each interaction and extract 'Description', 'Patient', and 'Doctor' parts
for curData in data:
    curData = curData.split('\n')
    pInd = (curData.index('Patient:'))
    dInd = (curData.index('Doctor:'))
    curQuery = curData[4].replace(',',';') + "."
    for j in range(pInd+1,dInd):
        curQuery += curData[j].replace(',',';')
    queries.append(curQuery)
    
    curResp=""
    for j in range(dInd+1,len(curData)):
        curResp += curData[j].replace(',',';')
    answers.append(curResp)


# Create a DataFrame and save it to a CSV file
df = pd.DataFrame({'Query': queries, 'Answer': answers})
df.to_csv(sys.argv[2], index=False)
