import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from urllib.request import urlopen

data = pd.read_csv('movies_tvshow.csv')


options = st.multiselect(
     'What tv shows do you like?',
     data["title"])
col1, col2 = st.columns(2)
   
@st.cache
def data_preprocessing(data):
    genres = pd.get_dummies(data["listed_in"])
    data_ext = data[["title", "release_year"]]
    train_data = pd.concat([data_ext, genres], axis = 1)
    train_data.set_index("title", inplace = True)
    return train_data

@st.cache
def nearest_neighbors(train_data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_data) 
    model = neighbors.NearestNeighbors(n_neighbors = 6)

    model.fit(scaled_data)

    dist, idlist = model.kneighbors(scaled_data)  
    return dist, idlist 


def recommending_per_watch_history(name, idlist):
    netflix_list = []
    netflix = data[data["title"] == name].index

    netflix = netflix[0]
    for id in idlist[netflix]:
        netflix_list.append(data.loc[id].title)
    if name in netflix_list:
        netflix_list.remove(name)
    with st.expander('Because you like {}, you might also like:'.format(name)):
        for i in netflix_list:
            st.write(i)
            st.video(data[data['title']==i]['trailer_link'].values[0])

           
            



if __name__ == "__main__":
    train_data = data_preprocessing(data)
    dist, idlist = nearest_neighbors(train_data)
    for show in options:
        recommending_per_watch_history(show, idlist)