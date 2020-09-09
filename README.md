# Recommender System for Amazon Fine Foods
Recommender Engine built with Surprise, focusing on item recommendation
Backend API for prediction provided

## To run the code
Make sure to have conda installed.
Run 
- `conda env create --file conda-env.txt`
- `python main.py`
- `uvicorn backend.app:app --reload --workers 1 --host 0.0.0.0 --port 8008`

Then visit http://localhost:8008

## Table of Contents
- [Background](#background)
- [Motivation](#motivation)
- [Data](#data)
    - [Understanding](#understanding)
    - [Preparation](#preparation)
- [Model](#model)
    - [KNN](#knn)
    - [SVD](#svd)
- [Application](#application)
- [Conclusion](#conclusion)
- [Next Step](#next-step)
- [Tools Used](#tools-used)

## Project directory
- images
    - images for the README
- backend
    - script for backend service
- model
    - data
        - directory for all the data used in the program
    - script for downloading, parsing, training,and exporting the model
    - notebook for visualization and notes while experimenting
    
## Background
Many services today thrive on recommendations for their users, Amazon is one of them. When browsing through products, we are often introduced to other products that might be similar to them in order to ease our shopping experience. 

## Motivation
I wanted to learn the troubles of building a recommender engine and also built an API service that could be used. 

## Data
Data was from downloaded from [SNAP](https://snap.stanford.edu/data/web-FineFoods.html#:~:text=Dataset%20information,from%20all%20other%20Amazon%20categories.). The timespan range from Oct 1999 to Oct 2012.

The data is not included in the repo, but a script is included to retrieve the file.

### Understanding
The data came in text format, require processing into one that the model will understand. A script has been written to parse the text file.

In total, there are
- 568454 reviews
- 256059 users
- 74258 prodcuts

During exploration of the data, I have found that majority of the users and items are rated less than 10 times. While majority of the ratings hover around a 4 or a 5. 

Distribution of User Ratings
![Distribution of User Ratings](/images/n_user_ratings.jpg)

Distribution of Item Ratings
![Distribution of Item Ratings](/images/n_item_ratings.jpg)

Distribution of Overall Ratings
![Distribution of Overall Ratings](/images/rating_dist.jpg)

Due to the sheer size of the data, the utility matrix would be around the size of 19 billion elements. Not a workable size for my personal computer. 

Based on the information found in the exploration of the data, I made a decision to look at items that have been rated more, this trims down the matrix and also allows for recommendation on items with more data. However this comes at a cost of losing out recommending items with lesser known information.

### Preparation
To prepare the data for modeling, the original data need to be in a format of 'user item rating' as required by the model.

Thus we need to parse the data out from text file into user item rating format.
Since exploration of the data was part of the steps, when I extracted the data I made it into a pandas dataframe for easier exploration. Conveniently, surprise models also allow for loading from pandas dataframe.

Overall steps were
- Parse data
- Prepare into user item rating format

## Model
For this engine, I have used two models: KNN and SVD.
KNN was used due to its similarity measures, allowing me to retrieve similar items. 
While SVD was used for user's item recommendations.

KFold cross validation was done with RMSE measure to evaluate the models. Both KNN and SVD were found to have similar rmse scores of around 0.6 to 0.7.

Finally randomize hyperparameter search was done to attempt to improve the scores.

Ideally, KNN could have done both user and item recommendations. However my personal laptop could not handle the KNN computations. 


### KNN
KNN is an algorithm that does prediction of based on nearest neighbors. To predict rating for a particular user, nearby neighbors are taken into account for its computation.


### SVD
SVD is a matrix factorization algorithm that factorize the original matrix (utility matrix) into three parts: user, eigen, and items. 

User matrix is a matrix of user with latent features while item matrix is a matrix of item with its latent features.

However the particular algorithm used was Simon Funk's version. Allowing us to estimate ratings for unknown parts of the matrix. 

Each of these models allow us to get recommendations based on similarity or latent features for either items similar to other items or items based on user's ratings of other items.

## Application
This engine allows us to retrieve recommendation of items. This can then be implemented as part of part of the backend to provide the recommendations.

I have written a backend API to showcase the recommendations, one issue I've encounter is the prediction speed which I have taken some measures to mitigate. 

## Conclusion
- Recommender engines often deal with large data troubles
- Matrix factorizations do not suffer as badly as KNN on memory issues
- Looking at the recommendations require domain knowledge to see if they're relevant because rmse only reflects upon how the user will rate and item not so much whether they will consume the product
- Backend API might face bottleneck issues when predicting

## Next Step
- Push data into a database so it's not part of the program
- Refractor codebase so code logic is more modular

## Tools Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Surprise
- FastAPI
- Uvicorn + Gunicorn
