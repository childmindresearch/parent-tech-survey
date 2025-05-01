import pandas as pd
import string
from embeddings import get_embedding

df = pd.read_csv('data/PPTOB_Data_03.21.csv')

cols = df.columns.to_list()

# Variables of interest:
# survey_num - Participant grouping, where 1= Group 1 (7-10), 2=Group 2 (11-14), 3 = Group 3 (15-18), and 4 = Group 4 (19-22)
# sm_def - Free text response of how would you define “social media”
# bereal:zoom - All the different platforms where participants rated if they thought the platform was social media or not
# q2_24 - Free text response of what factors do you consider when determining if content is “age-appropriate”
# q2_23a:g - Matrix of factors that parents might consider when determining if content is age appropriate

factors_matrix = ['q2_23_' + str(x) for x in list(string.ascii_lowercase)[0:7]]
website_matrix = cols[cols.index('bereal') :cols.index('zoom') + 1]

subset_cols = ['record_id', 'survey_num', 'sm_def', 'q2_24'] + factors_matrix + website_matrix
df = df[subset_cols]


# embeddings 
df['q1_embedding'] = df['sm_def'].apply(lambda x: get_embedding(x))
df['q2_embedding'] = df['q2_24'].apply(lambda x: get_embedding(x))

# save embeddings 
df.to_csv('data/text_response_embeddings.csv', index=False)