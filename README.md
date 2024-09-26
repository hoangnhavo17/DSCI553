# Yelp Recommendation System

## Overview
This project is a Yelp Recommendation System built using Python. It aims to provide personalized restaurant or business recommendations based on user preferences, reviews, and ratings from Yelp. By leveraging data from Yelp, the system employs algorithms that analyze user input to recommend the best options.

## Features
- User-based Collaborative Filtering: Recommends businesses by identifying users with similar preferences.
- Content-based Filtering: Recommends businesses by matching user preferences with business attributes.
- Hybrid Model: Combines collaborative filtering and content-based methods for improved recommendations.

## Method Description:
Minor adjustments made to the item-based collaborative filtering were to obtain the weights of other items with a similarity greater than 0.9. With this, I used the number of neighbors to determine the item-based prediction's weight on the final prediction.

To train the model, I developed features from the provided JSON files:

user.json:
- average_stars: average rating by user
- review_count: number of reviews made by user
- yelping_days: number of days since the user created their Yelp account

business.json:
- stars: business's overall or average rating
- review_count: number of reviews made to business
- is_open: whether the business is still open or not(at least this is my interpretation of this attribute)

review_train.json business ids:
- useful_avg: average useful reactions
- funny_avg: average funny reactions
- cool_avg: average cool reactions

Short on time, unable to utilize the following features derived from review_train.json. With more time, I would like to see its effects on training and then tune the parameters accordingly.

For both business IDs and user IDs:
- min: minimum rating 
- max: maximum rating
- std_dev: standard deviation of ratings
- median: median rating
- skew: asymmetry of ratings
- kurtosis: tailedness of rating distribution
- consistency: ratio of standard deviation to mean of ratings, measuring consistency of ratings
- mode: most frequent rating

photo.json:
- photo_count: number of photos affiliated with the business

checkin.json:
- total_checkins: total check-ins overall for each business
- avg_checkin_per_day: average check-ins per day for each business
- peak_day: day of the week with the most check-ins
- peak_hour: hour of the day with the most check-ins

tip.json:
    
For user ID:
- user_freq: how frequently does the user leave tips
- user_rec: how recently did the user leave a tip

For business ID:
- bus_vol: total volume of tips for the business
- bus_uu: number of unique users that have left a tip for the business

For the model parameters, I did not have the time needed to run cross-validation to find the most optimal parameters. Thus, I tested with typical values for each parameter and began adjusting through small increments and decrements to reduce my RMSE.

## Error Distribution:
>=0 and <1: 102059
>=1 and <2: 32983
>=2 and <3: 6216
>=3 and <4: 785
>=4: 1

## RMSE:
0.98017

## Execution Time:
602s

## Final Results
The outcome of the project with the test set resulted in an RMSE below 0.98, surpassing the TA's RMSE threshold and achieving full points for the project.
