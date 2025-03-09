import pandas as pd

# Load data from JSON files
tracks_df = pd.read_json('tracks.json')
authors_df = pd.read_json('authors.json')

'''
  track_id   title  likes  clicks  full_listens author_id
0      001  Song A    150     300           120        A1
1      002  Song B    200     400           180        A2
2      003  Song C    100     250            95        A3
'''

'''
  author_id     name  author_listeners genre
0        A1  Artist X             5000   Rock
1        A2  Artist Y             8000    Pop
2        A3  Artist Z             3000   Jazz
'''

# Merge the dataframes on the common 'author_id' field
merged_df = pd.merge(tracks_df, authors_df, on='author_id', how='inner')

'''
  track_id   title  likes  clicks  full_listens author_id     name  author_listeners genre
0      001  Song A    150     300           120        A1  Artist X             5000   Rock
1      002  Song B    200     400           180        A2  Artist Y             8000    Pop
2      003  Song C    100     250            95        A3  Artist Z             3000   Jazz
'''

# Select relevant content features
content_features = ["likes", "clicks", "full_listens", "author_listeners", "genre"]
content_features_df = merged_df[content_features]

# Display the content features dataset
print(content_features_df)

'''
   likes  clicks  full_listens  author_listeners genre
0    150     300           120             5000   Rock
1    200     400           180             8000    Pop
2    100     250            95             3000   Jazz
'''
