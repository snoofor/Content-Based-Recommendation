############################################################
# IMDB top100 movies web-scrapping
# Movie Suggestion Cosine Similarities by using Content Based Recommendation
############################################################

import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.expand_frame_repr", False)

############################################################

url = "https://www.imdb.com/list/ls006266261/"

response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

# after this decide what do you want
#   then go to website and right click and go to inspect

############################################################
movie_name = []
year = []
time = []
rating = []
meta_score = []
votes = []
gross = []
description = []
len(description)
############################################################
# document.querySelector("#main > div > div.lister.list.detail.sub-list > div.lister-list > div:nth-child(1)")
#   copy js path for each list main list

movie_data = soup.findAll('div', attrs={'class': 'lister-item mode-detail'})

for store in movie_data:
    name = store.h3.a.text
    movie_name.append(name)
    run_time = store.find("span", class_="runtime").text.replace(" min", "")
    time.append(run_time)
    year_of_release = store.find("span", class_="lister-item-year text-muted unbold").text.\
        replace("(", "").replace(")", "")
    year.append(year_of_release)
    year = list(map(lambda x: x.replace('I 2011', '2011'), year))
    rate = store.find("div", class_="ipl-rating-star small").text.replace("\n", "")
    rating.append(rate)
    meta = store.find("span", class_="metascore").text.replace(" ", "")\
        if store.find("span", class_="metascore") else "NaN"
    meta_score.append(meta)
    value = store.find_all("span", attrs={"name": "nv"}) # 0 index is vote and # 1 index is gross
    vote = value[0].text
    votes.append(vote)
    grosses = value[1].text if len(value) > 1 else ""
    gross.append(grosses)
    descript = store.find("p", class_="").text.replace("\n", "")
    description.append(descript)

############################################################
# Create DataFrame

movie_DF = pd.DataFrame({"Title": movie_name,
                         "Year": year,
                         "Time": time,
                         "Rate": rating,
                         "MetaScore": meta_score,
                         "Vote": votes,
                         "Gross": gross,
                         "Description": description
                         })

# There are no NaN Value ind DataFrame
movie_DF.isnull().any()

movie_DF.to_csv("top100-imdb.csv")

############################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_ = pd.read_csv("Web-Scrapping/top100-imdb.csv", low_memory=False)

df = df_.copy()
df.describe().T

df.columns

del df["Unnamed: 0"]

df[0:1]

df[df["Title"].str.contains("The Godfather")]

"""indices1 = pd.Series(df.index, index=df["title"])
type(indices1)
del indices1"""


def similarity_p_prep(dataframe, lang=None, variable=None):
    tfID = TfidfVectorizer(stop_words=lang)
    dataframe[variable] = dataframe[variable].fillna("")
    tfID_matrix = tfID.fit_transform(df[variable])
    # cosine_sim = cosine_similarity(tfID_matrix, tfID_matrix)
    return tfID_matrix


def cos_sim():
    cosine_sim = cosine_similarity(tfID_matrix, tfID_matrix)
    return cosine_sim


"""# stop_words egnlish ingilizce yaygın kullanılan anlam ifade etmeyen kelimeleri almaz
#   and in on gibi, çok fazla boş gözlem değeri gelmesin

tfID = TfidfVectorizer(stop_words="english")

df["overview"].isnull().sum()

# overview NaN olan değerleri sil veya boşluk ile değiştir
df["overview"] = df["overview"].fillna("")

tfID_matrix = tfID.fit_transform(df["overview"])

tfID_matrix.shape

tfID.get_feature_names()

tfID_matrix.toarray()

########################################################################
# Cosine Similarity

cosine_sim = cosine_similarity(tfID_matrix, tfID_matrix)
# 1 index teki filmin bütün diğer filmler ile benzerlik skoru var
cosine_sim[1]"""


tfID_matrix = similarity_p_prep(df, "english", "Description")
cosine_sim = cos_sim()


def cbr(title, cosine_sim, dataframe, variable=None, count=10):
    # create index
    indices = pd.Series(dataframe.index, index=dataframe[variable])
    indices = indices[~indices.index.duplicated(keep="last")]
    # hold index of title
    movie_index = indices[title]
    # calculate similarities scores by titles
    similarity_score = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # first 10 movie except itself
    movie_indices = similarity_score.sort_values("score", ascending=False)[1:count].index
    return dataframe[variable].iloc[movie_indices]


movie_S_TGF = cbr("The Godfather", cosine_sim, df, "Title")














