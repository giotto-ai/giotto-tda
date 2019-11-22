# football-tda
The purpose of this project is to show a possible application of TDA. Our use case is 
based on football and the goal (pun intended) is to try to forecast the outcome of a
match. 

You can find our blog post at this 
[link](https://towardsdatascience.com/the-shape-of-football-games-1589dc4e652a).

## Data
The dataset we used can be found [here](https://www.kaggle.com/hugomathien/soccer).
It is a collection of more than 25,000 european football matches from 2008 to 2016. 
For each match, the starting eleven are available for both teams, as well as match 
statistics and bookmaker odds. 

The dataset also contains the attributes of more than 10,000 players taken from EA 
Sports' FIFA video game series, including weekly updates. 

## Feature Creation
The assumption we made is that each match can be modelled as the attributes of the 
starting eleven of the two teams. Since in this way the number of features was too 
high, an additional aggregation step was required (see the notebook for further 
details). 

Thus, each match can be considered as a vector in a vector space and the totality of 
matches can be viewed as a point cloud.

For capturing local information surrounding a match, we computed persistent homology 
of its k-nearest neighbours and use it as a feature. 

## Model
We cross-validated a random forest classifier and train it to predict the outcome of 
a match. In order to validate our results, we used an elo-rating system and the odds 
of the market as baselines. 

## Results
Our results show that our model out-performs the elo-rating system and is 
comparable to the market.  

## Notebook overview
Given the promising results, we tried to simulate an entire championship with the 
ultimate purpose of evaluating the impact that a player would have had if hired by 
our favorite team. Therefore, we offer the possibility to select both the favorite 
player and the lucky team where to insert him. Then you can simulate the championship
and check if your player improves the final ranking of his new team (little spoiler: 
Messi does!). 

Enjoy!

## Requirements
In order to run the notebook, the following python packages are required: 

- giotto-learn 0.1.2
- pandas 0.25.3
- pyarrow 0.15.1
- tqdm 4.38.0
- wget 3.2  
