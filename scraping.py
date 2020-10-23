import requests
#import time
import os
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from pyfiglet import Figlet
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import RandomOverSampler

nlp = spacy.load('en_core_web_md')

fig = Figlet(font='slant')

base_url = 'https://www.lyrics.com'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

columns = ['artist', 'title', 'lyrics']


def format_artist_name(name: str) -> str:
  '''format the artists name

  Parameter
  ---------
  name : str
    name of an artist

  Returns
  -------
  str
    name of the artist in lower case and without spaces
  '''
  return name.lower().replace(' ', '-')

def get_artist_url(name: str) -> str:
  ''' get the url for the artist page

  Parameter
  ---------
  name : str
    name of the artist already cleaned up with format_artist_name

  Returns
  -------
  str
    url to the artist page
  '''
  name = format_artist_name(name)
  return f'{base_url}/artist.php?name={name}&o=1'

def clean_title(title: str) -> str:
  '''clean up the song title

  Parameter
  ---------
  title: str
    title of the song

  Returns
  -------
  str
    title without access spaces and without information in brackets.
  '''
  return title[:title.find('[')].strip().lower()


def get_artist_link_list(name:str) -> [str, [str]]:
  ''' request the artist page return a list of links

  Parameters
  ----------
  name
    name of the artist you want to scrape

  Returns
  -------
  [name, [songtitle, link]]

  name : str
    name of the Artist in the correct capitalization
  songtitle : str
    title of the song
  link:
    link to the song (will be relative)
  '''
  artist_page = requests.get(get_artist_url(name), headers = headers)

  artist_soup = BeautifulSoup(artist_page.text, 'lxml')
  list_of_links = []
  # get all the links
  old_title = None
  # check if there is no artist of that name
  if not artist_soup.find('h1', {'class': 'artist'}):
    return [None, None]
  artist_name = artist_soup.find('h1', {'class': 'artist'}).get_text()

  for a in artist_soup.find_all('a'):
    if a.get('href'):
      if '/lyric/' in a.get('href'):
        title = a.get_text()
        cleaned_title = clean_title(title)
        if old_title != cleaned_title:
          list_of_links.append([cleaned_title, a.get('href')])
          old_title = cleaned_title
  return [artist_name, list_of_links]


def get_lyrics_url(link):
  ''' process link into a url

  Parameter
  ---------
  link : str
    relative link to the lyrics page

  Return
  ------
  link : str
    absolute url to the lyrics page
  '''
  return f'{base_url}{link}'

def get_lyrics(link):
  '''scrape lyrics

  Parameter
  ---------
  link : str
    relative link to the lyrics page

  Return
  ------
    lyrics : str
    songtext scraped from lyrics.com
  '''
  lyrics_page = requests.get(get_lyrics_url(link), headers)
  lyrics_soup = BeautifulSoup(lyrics_page.text, 'lxml')
  return lyrics_soup.body.pre.get_text()

# Define cleaning function
def clean_text(review, model):
    """preprocess a string (tokens, stopwords, lowercase, lemma & stemming) returns the cleaned result
        params: review - a string
                model - a spacy model

        returns: list of cleaned tokens
    """

    new_doc = ''
    doc = model(review)
    for word in doc:
        if not word.is_stop and word.is_alpha:
            new_doc = f'{new_doc} {word.lemma_.lower()}'

    return new_doc

def evaluate_model(X_train, y_train, X_test, y_test):
  ''' print evaluation of the model

  Parameter
  ---------
  X_train : Pandas DataFrame
    Training Data X
  y_train : Pandas Series
    Traning Data y
  X_test : Pandas DataFrame
    Test Data X
  y_test : Pandas Series
    Test Data y

  Return
  ------
  None
  '''
  print(f'Evaluation')
  print('-----------')

  pipeline.fit(X_train['lyrics'], y_train)
  print('Trainingsscore: ', pipeline.score(X_train['lyrics'],y_train))
  print('Testscore:      ', pipeline.score(X_test['lyrics'], y_test))


print(fig.renderText('Song Scraper'))
print('\n\nby Leli\n\n')

n = input('how many artists do you want to compare?\n')

# check if the input is valid
while type(n) != int:
  try:
    # can it be converted to int?
    n = int(n)
    # if it is an integer, is it large enough?
    if n < 2:
      n = input('\nYou need at least 2 artists to be able to compare anything.3\n')
  except:
    n = input(f'\nYour input "{n}" does not seem to be an intereger.\nPlease enter an integer value for how many artists you want to compare.\n')
print()
all_the_lyrics = pd.DataFrame(columns=columns)

for artist_number in range(n):
  artist = input(f'\nWhat is the {artist_number+1}. artists name?\n')

  artist_name, list_of_links = get_artist_link_list(artist)
  while list_of_links is None or list_of_links == []:
    artist = input(f'\nSorry but I can\'t find this artist.\nCould you check the spelling and try again?\n')
    artist_name, list_of_links = get_artist_link_list(artist)

  file = f'{format_artist_name(artist)}.csv'
  print(fig.renderText(artist_name))
  if not os.path.exists(file):
    artist_df = pd.DataFrame(columns=columns)
    print(f'\n{artist_name} is not yet scraped. This may take a while.\n')


    for i in tqdm(range(len(list_of_links))):
      song, link = list_of_links.pop()
      artist_df.loc[i] = [artist_name, song, get_lyrics(link)]
    pd.DataFrame.to_csv(artist_df, file)

  else:
    print(f'\nYou are lucky!\n{artist_name} has already been scraped!\n')
    artist_df = pd.read_csv(file, index_col=0)
  all_the_lyrics = all_the_lyrics.append(artist_df, ignore_index=True )



# do the maschine learning stuff

all_the_lyrics['lyrics'] = all_the_lyrics['lyrics'].apply(clean_text, model = nlp)


X = all_the_lyrics[['lyrics']]
y = all_the_lyrics['artist']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# TO-DO
# - make some usefull predictions
# - don't scrape the whole site
#   - did make it slower actually so i changed it back

pipeline = Pipeline([
  ('vectorize', TfidfVectorizer()),
  ('naive bayes', MultinomialNB())
])

ros = RandomOverSampler()

X_ros, y_ros = ros.fit_resample(X_train, y_train)
evaluate_model(X_ros, y_ros, X_test, y_test)

print(fig.renderText('Prediction Time'))

print('\nnow let\'s make a prediction\n')
print('If you want to stop, enter \'!q\'\n')
to_predict = input('Give me some text\n')

while to_predict != '!q':
  test_the_text = pd.DataFrame([to_predict], columns=['lyrics'])
  print(pipeline.predict(test_the_text['lyrics'])[0])
  prediction = pipeline.predict(test_the_text['lyrics'])[0]
  print()
  print(fig.renderText(prediction))
  print(f'I predict this is {prediction}')
  to_predict = input('Give me another one.\n')
