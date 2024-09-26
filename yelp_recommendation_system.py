"""
Method Description:
Minor adjustments made to the item-based collaborative filtering was to obtain the weights of other items with similarity greater than 0.9. With this, I used the number of neighbors to determine the item-based prediction's weight on the final prediction.

To train the model, I developed features from the provided json files:
user.json:
    - average_stars: average rating by user
    - review_count: number of reviews made by user
    - yelping_days: number of days since user created their yelp account

business.json:
    - stars: business's overall or average rating
    - review_count: number of reviews made to business
    - is_open: whether the business is still open or not(at least this is my interpretation of this attribute)

review_train.json:
    For business ids:
        - useful_avg: average useful reactions
        - funny_avg: average funny reactions
        - cool_avg: average cool reactions

    Short on time, unable to utilize the following features derived from review_train.json.
    With more time, I would like to see its effects on training and then tune the parameters accordingly.
    For both business ids and user ids:
        - min: minimum rating 
        - max: maximum rating
        - std_dev: standard deviation of ratings
        - median: median rating
        - skew: assymetry of ratings
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
    For user id:
        - user_freq: how frequently does the user leave tips
        - user_rec: how recentyl did the user leave a tip
    For business id:
        - bus_vol: total volume of tips for the business
        - bus_uu: number of unique users that have left a tip for the business

For the model parameters, I did not have the time needed to run cross validation to find the most optimal parameters. Thus, I tested with typical values for each parameter and began adjusting through small increments and decrements to reduce my RMSE.

Error Distribution:
>=0 and <1: 102059
>=1 and <2: 32983
>=2 and <3: 6216
>=3 and <4: 785
>=4: 1

RMSE:
0.98017

Execution Time:
602s

"""
from pyspark import SparkContext
import sys
import time
import json
import numpy as np
from xgboost import XGBRegressor
import math
from datetime import datetime
from collections import defaultdict, Counter
from scipy.stats import skew, kurtosis

def parse_row(row):
    columns = row.split(',')
    user_id, business_id, stars = columns
    return user_id, business_id, float(stars)

def calc_sim(bus1, bus2):
    if bus1 not in business_baskets_dict.keys() or bus2 not in business_baskets_dict.keys():
        return 0.0
    
    users_common = business_baskets_dict[bus1] & business_baskets_dict[bus2]
    
    if len(users_common) <= 1:
        weight = abs(business_avg_dict[bus1] - business_avg_dict[bus2])
        return (5.0 - weight) / 5.0
    elif len(users_common) == 2:
        bus1_ratings = [float(bur_dict[(bus1, user)]) for user in users_common]
        bus2_ratings = [float(bur_dict[(bus2, user)]) for user in users_common]
        weight = sum(abs(r1 - r2) for r1, r2 in zip(bus1_ratings, bus2_ratings)) / 2.0
        return (5.0 - weight) / 5.0
    else:
        bus1_ratings = [float(bur_dict[(bus1, user)]) for user in users_common]
        bus2_ratings = [float(bur_dict[(bus2, user)]) for user in users_common]
        
        avg_bus1_ratings = sum(bus1_ratings) / len(bus1_ratings)
        avg_bus2_ratings = sum(bus2_ratings) / len(bus2_ratings)
        
        covariance = sum((r1 - avg_bus1_ratings) * (r2 - avg_bus2_ratings) for r1, r2 in zip(bus1_ratings, bus2_ratings))
        std_dev_bus1 = sum((r - avg_bus1_ratings) ** 2 for r in bus1_ratings) ** 0.5
        std_dev_bus2 = sum((r - avg_bus2_ratings) ** 2 for r in bus2_ratings) ** 0.5

        if std_dev_bus1 == 0 or std_dev_bus2 == 0:
            return 0.0
        else:
            return covariance / (std_dev_bus1 * std_dev_bus2)
    
def item_based_prediction(bus, user):
    if user not in user_baskets_dict:
        return avg_rating, 0
    if bus not in business_baskets_dict:
        return user_avg_dict[user], 0
    
    similarities = list()
    
    for rated_business in user_baskets_dict[user]:
        sim = calc_sim(bus, rated_business)
        if sim > 0.9:
            rating = float(bur_dict[(rated_business, user)])
            similarities.append((sim, rating))
    
    top_neighbors = sorted(similarities, key = lambda x: x[0], reverse=True)
    
    weighted_sum = sum(sim * rating for sim, rating in top_neighbors)
    total_weight = sum(abs(sim) for sim, _ in top_neighbors)
        
    if total_weight == 0.0:
        return avg_rating, len(top_neighbors)
    else:
        return weighted_sum / total_weight, len(top_neighbors)

def calc_mode(ratings):
    if not ratings:
        return None
    
    counts = Counter(ratings)
    max_count = max(counts.values())
    mode = [k for k, v in counts.items() if v == max_count]
    
    return mode[0]
    
def yelping_days(yelping_since):
    yelping_date = datetime.strptime(yelping_since, '%Y-%m-%d')
    days_since = (datetime.today() - yelping_date).days
    return days_since

def checkin_data(time_data):
    days_mapping = {
        'Mon': 1,
        'Tue': 2,
        'Wed': 3,
        'Thu': 4,
        'Fri': 5,
        'Sat': 6,
        'Sun': 7
    }
    
    day_counts = defaultdict(int)
    hour_counts = defaultdict(int)
    
    for key, count in time_data.items():
        day_str, hour_str = key.split('-')
        day = days_mapping.get(day_str, 0)
        hour = int(hour_str) if hour_str.isdigit() else 0
        day_counts[day] += int(count)
        hour_counts[hour] += int(count)
        
    days_filtered = {k: v for k, v in day_counts.items() if k != 0}
    
    total_checkins = sum(days_filtered.values())
    avg_checkins_per_day = sum(days_filtered.values()) / len(days_filtered) if days_filtered else 0
    
    peak_day = max(days_filtered, key=days_filtered.get, default=None)
    peak_hour = max(hour_counts, key=hour_counts.get, default=None)
    
    return (int(total_checkins), float(avg_checkins_per_day), peak_day, peak_hour)
    
def calc_recency(latest_date_str):
    today = datetime.today()
    latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d")
    recency_days = (today - latest_date).days
    return recency_days
    
def prepare_features(data):
    features = list()
    for row in data:
        user, bus, *rating = row
        user_features = list(user_dict.get(user, [None, None, None]))
        business_features = list(business_dict.get(bus, [None, None, None]))        
        #bus_review_features = list(bus_review_dict.get(bus, [None, None, None, None, None, None, None, None]))
        bus_react_features = list(bus_react_dict.get(bus, [None, None, None]))
        #user_review_features = list(user_review_dict.get(user, [None, None, None, None, None, None, None, None]))
        photo_count = photo_dict.get(bus, None)
        photo_feature = [photo_count]
        checkin_features = list(checkin_dict.get(bus, [None, None, None, None]))
        user_tips_features = list(user_tips_dict.get(user, [None, None]))
        bus_tips_features = list(bus_tips_dict.get(bus, [None, None]))
        
        #current_features = user_features + business_features + bus_review_features + bus_react_features + user_review_features + photo_feature + checkin_features + user_tips_features + bus_tips_features
        current_features = user_features + business_features + bus_react_features + photo_feature + checkin_features + user_tips_features + bus_tips_features

        features.append(current_features)
    
    return np.array(features, dtype='float32')

def write_output(output_file_name, test_pairs, predictions):
    with open(output_file_name, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for (user, bus), pred in zip(test_pairs, predictions):
            f.write(f'{user},{bus},{pred}\n')

def calc_rmse(predictions, test):
    squared_errors = 0.0
    num_predictions = 0
    
    pred_dict = {(user, business): prediction for user, business, prediction in predictions}
    
    for user, business, actual_rating in test:
        if (user, business) in pred_dict:
            predicted_rating = pred_dict[(user, business)]
            
            squared_errors += (predicted_rating - actual_rating) ** 2
            num_predictions += 1
            
    mse = squared_errors / num_predictions
    
    rmse = math.sqrt(mse)
    return rmse
            
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit task2_3.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)

    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    sc = SparkContext(appName="task2_3")
    sc.setLogLevel("ERROR")
    
    start_time = time.time()
    train_file = sc.textFile(folder_path + '/yelp_train.csv')
    train_head = train_file.first()
    train_file = train_file.filter(lambda x: x != train_head)
    
    train = train_file.map(parse_row)

    business_baskets = train.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)
    business_baskets_dict = {bus: users for bus, users in business_baskets.collect()}

    user_baskets = train.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)
    user_baskets_dict = {user: buses for user, buses in user_baskets.collect()}
    
    business_avg = train.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x))
    business_avg_dict = {bus: avg for bus, avg in business_avg.collect()}
    
    user_avg = train.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x))
    user_avg_dict = {user: avg for user, avg in user_avg.collect()}
    
    bur = train.map(lambda x: ((x[1], x[0]), x[2]))
    bur_dict = {bu: r for bu, r in bur.collect()}
    
    avg_ratings = business_avg.map(lambda x: x[1])
    avg_rating = avg_ratings.mean()

    user_file = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], (float(x['average_stars']), int(x['review_count']), int(yelping_days(x['yelping_since'])))))
    user_dict = user_file.collectAsMap()

    business_file = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (float(x['stars']), int(x['review_count']), int(x['is_open']))))
    business_dict = business_file.collectAsMap()
        
    review_file = sc.textFile(folder_path + '/review_train.json').map(lambda x: json.loads(x))
    
    #bus_review = review_file.map(lambda x: (x['business_id'], x['stars']))
    #bus_stats = bus_review.groupByKey().mapValues(list).map(lambda x: (x[0], (min(x[1]), max(x[1]), np.std(x[1]), np.median(x[1]), skew(x[1]), kurtosis(x[1]), np.std(x[1])/np.mean(x[1]), calc_mode(x[1]))))
    #bus_review_dict = bus_stats.collectAsMap()
    
    bus_react = review_file.map(lambda x: (x['business_id'], (float(x['useful']), float(x['funny']), float(x['cool']))))
    react_avg = bus_react.aggregateByKey((0,0,0,0), lambda acc, value: (acc[0] + value[0], acc[1] + value[1], acc[2] + value[2], acc[3] + 1), lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1], acc1[2] + acc2[2], acc1[3] + acc2[3])).mapValues(lambda sum_counts: (sum_counts[0]/sum_counts[3], sum_counts[1]/sum_counts[3], sum_counts[2]/sum_counts[3]))
    bus_react_dict = react_avg.collectAsMap()
    
    #user_review = review_file.map(lambda x: (x['user_id'], x['stars']))
    #user_stats = user_review.groupByKey().mapValues(list).map(lambda x: (x[0], (min(x[1]), max(x[1]), np.std(x[1]), np.median(x[1]), skew(x[1]), kurtosis(x[1]), np.std(x[1])/np.mean(x[1]), calc_mode(x[1]))))
    #user_review_dict = user_stats.collectAsMap()

    photo_dict = sc.textFile(folder_path + '/photo.json').map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['photo_id'])).distinct().countByKey()
    
    checkin_file = sc.textFile(folder_path + '/checkin.json').map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (checkin_data(x['time']))))
    checkin_dict = checkin_file.collectAsMap()
    
    tip_file = sc.textFile(folder_path + '/tip.json').map(lambda x: json.loads(x))
    
    user_freq = tip_file.map(lambda x: (x['user_id'], x['business_id'])).distinct().countByKey()
    user_rec = tip_file.map(lambda x: (x['user_id'], x['date'])).reduceByKey(lambda date1, date2: max(date1, date2)).mapValues(calc_recency)
    user_tips = user_rec.map(lambda x: (x[0], (int(x[1]), int(user_freq.get(x[0], 0)))))
    user_tips_dict = user_tips.collectAsMap()
    
    bus_vol = tip_file.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda a,b: a+b)
    bus_uu = tip_file.map(lambda x: (x['business_id'], x['user_id'])).distinct().countByKey()
    bus_tips = bus_vol.map(lambda x: (x[0], (int(x[1]), int(bus_uu.get(x[0], 0)))))
    bus_tips_dict = bus_tips.collectAsMap()
    
    train_data = train.collect()
    x_train = prepare_features(train_data)
    y_train = np.array([row[2] for row in train_data], dtype='float32')
    
    test_file = sc.textFile(test_file_name)
    test_head = test_file.first()
    test_file = test_file.filter(lambda x: x != test_head)
    
    val = test_file.map(parse_row)
    test = val.map(lambda x: (x[0], x[1]))
    test_pairs = [(x[0], x[1]) for x in test.collect()]
    
    item_pred = list()
    item_neighbors = list()
    for pair in test_pairs:
        prediction, neighbors = item_based_prediction(pair[1], pair[0])
        prediction = min(5.0, prediction)
        prediction = max(1.0, prediction)
        item_pred.append(prediction)
        item_neighbors.append(neighbors)
        
    x_val = prepare_features(test.collect())
    
    params = {
        'booster': 'gbtree',
        'colsample_bytree': 0.5, 
        'learning_rate': 0.04, 
        'max_depth': 12, 
        'min_child_weight': 100, 
        'n_estimators': 300, 
        'random_state': 901, 
        'subsample': 0.7
    }
    
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    model_pred = model.predict(x_val)
    model_pred = np.clip(model_pred, 1.0, 5.0)

    hybrid_pred = list()
    max_neighbors = max(item_neighbors)
    for i in range(0, len(test_pairs)):
        alpha = item_neighbors[i] / max_neighbors 
        hybrid = (1 - alpha) * model_pred[i] + alpha * item_pred[i]
        hybrid_pred.append(hybrid)

    write_output(output_file_name, test_pairs, hybrid_pred)
    
    predicted_pairs = [(user, bus, pred) for (user, bus), pred in zip(test_pairs, hybrid_pred)]

    end_time = time.time()
    runtime = int(end_time - start_time)
    print(f"Duration: {runtime}")
    
    #rmse = calc_rmse(predicted_pairs, val.collect())
    #print(f"RMSE: {rmse}")
    
    sc.stop()