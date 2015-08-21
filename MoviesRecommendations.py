
# coding: utf-8

# # __Movie Recommendation System__
# Samuel Sherman

# Using ratings and movie id datasets from the movieLens webiste (http://grouplens.org/datasets/movielens/), I will apply model to predict movie ratings using the pyspark API and the Alternating Least Squares algorithm.

# In[5]:

import sys
import os

baseDir = os.path.join('data')
inputPath = os.path.join('movielens', 'als', 'data')

ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')


# I first define two function to parse the data, one for the movies and one for the user ratings.

# In[7]:

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)

def get_ratings_tuple(entry): # UserID::MovieID::Rating::Timestamp to (UserID, MovieID, Rating)
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry): # MovieID::Title::Genres to (MovieID, Title)
    items = entry.split('::')
    return int(items[0]), items[1]


ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print 'There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount)
print 'Ratings: %s' % ratingsRDD.take(3)
print 'Movies: %s' % moviesRDD.take(3)


# In[8]:

def sortFunction(tuple):
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)


# I will now create a function that takes in a tuple of the movie id and ratings and returns the movie id (key) and the number of ratings and the average rating (value).

# In[9]:

def getCountsAndAverages(IDandRatingsTuple): # Get num ratings and average rating
    MovieID = IDandRatingsTuple[0]
    numRatings = len(IDandRatingsTuple[1])
    averageRating = sum(IDandRatingsTuple[1])/float(numRatings)
    return (MovieID, (numRatings, averageRating))


# Here, I take the ratings RDD, which consists of a User Id, Movie Id, and Rating, and I extract the Movie Id and Rating. I then group by key, which will produce a tuple consisting of an iterable of ratings per movie id. Using the function defined above, I produce a new tuple with the Movie Id and the average rating. Finally, I join this RDD with the movies RDD and extract the Average Rating, the Movie Name, and the number of Ratings.

# In[10]:

# (MovieID, iterable of Ratings for that MovieID)
movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda (a,b,c): (b,c)).groupByKey())
print 'movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3)

# (MovieID, (number of ratings, average rating))
movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)
print 'movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3)

# (average rating, movie name, number of ratings)
movieNameWithAvgRatingsRDD = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda (a, (b, (c, d))): (d, b, c)))
print 'movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3)


# I filter out only the ratings with above 500 reviews and sort descending by each rating. The top 20 ratings are printed below.

# In[11]:

# Highest rating first
movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda (a,b,c): c > 500)
                                    .sortBy(sortFunction, False))
print 'Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20)


# I perform a random split on the ratings RDD to produce three datasets for training, validation, and testing (60%, 20%, 20%).

# In[12]:

trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)

print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                    validationRDD.count(),
                                                    testRDD.count())
print trainingRDD.take(3)
print validationRDD.take(3)
print testRDD.take(3)


# The following function will produce the root mean squared error.

# In[13]:

import math

def computeError(predictedRDD, actualRDD): # RMSE
    predictedReformattedRDD = predictedRDD.map(lambda (UserID, MovieID, Rating): ((UserID, MovieID), Rating)) 
    actualReformattedRDD = actualRDD.map(lambda (UserID, MovieID, Rating): ((UserID, MovieID), Rating)) 
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                        .map(lambda (k,v): (v[0]-v[1])**2))
    totalError = squaredErrorsRDD.sum()
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(totalError/float(numRatings))


# When applying the model I iterate of different values for the rank to determine which value will produce the lowest RMSE.

# In[14]:

from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda (a,b,c): (a,b))

seed = 5L
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank


# Using rank 8, from above, I apply the model using MLlib's ALS algorithm.

# In[15]:

myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda (a,b,c): (a,b))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE


# In[16]:

trainingAvgRating = trainingRDD.map(lambda x: x[2]).reduce(lambda a,b: a+b)/trainingRDD.count()
print 'The average rating for movies in the training set is %s' % trainingAvgRating

testForAvgRDD = testRDD.map(lambda (a,b,c): (a,b,trainingAvgRating))
testAvgRMSE = computeError(testRDD, testForAvgRDD)
print 'The RMSE on the average set is %s' % testAvgRMSE


# In[17]:

print 'Most rated movies:'
print '(average rating, movie name, number of reviews)'
for ratingsTuple in movieLimitedAndSortedByRatingRDD.take(50):
    print ratingsTuple


# Using some of the movies defined above, of which I have seen, I apply my own ratings and add them to the original training RDD. Using this data, I reapply the model.

# In[18]:

myUserID = 0

myRatedMovies = [
     (myUserID, 1088, 5),
	 (myUserID, 1047, 3.5),
	 (myUserID, 831, 5),
	 (myUserID, 1447, 4),
	 (myUserID, 1248, 5),
	 (myUserID, 587, 3),
	 (myUserID, 759, 2),
	 (myUserID, 1337, 3.5),
	 (myUserID, 1250, 5),
	 (myUserID, 1438, 4),
	 (myUserID, 1039, 5),
	 (myUserID, 811, 3),
	 (myUserID, 1775, 4),
	 (myUserID, 744, 5),
	 (myUserID, 983, 3),
	 (myUserID, 516, 2)
    ]

myRatingsRDD = sc.parallelize(myRatedMovies)
print 'My movie ratings: %s' % myRatingsRDD.take(10)


# In[19]:

trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)

print ('The training dataset now has %s more entries than the original training dataset' %
       (trainingWithMyRatingsRDD.count() - trainingRDD.count()))


# In[20]:

myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)


# Finally, I use the model above to predict rating for movies I have not rated, or seen, and print the top 20 recommended movies that are most appealing based on my ratings above.

# In[21]:

predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
testRMSEMyRatings = computeError(testRDD, predictedTestMyRatingsRDD)
print 'The model had a RMSE on the test set of %s' % testRMSEMyRatings


# In[22]:

myUnratedMoviesRDD = (moviesRDD
                      .map(lambda (x,y): (myUserID,x))
                      .filter(lambda x: x[1] not in [i[1] for i in myRatedMovies]))

predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)


# In[23]:

movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda (MovieID, (numRatings, averageRating)): (MovieID, numRatings))
predictedRDD = predictedRatingsRDD.map(lambda (user, MovieID, predictRating): (MovieID, predictRating))
predictedWithCountsRDD  = (predictedRDD
                           .join(movieCountsRDD))
ratingsWithNamesRDD = (predictedWithCountsRDD
                       .join(moviesRDD)
                       .filter(lambda (movieID, ((predictRating, numRatings), movieName)): numRatings > 75)
                       .map(lambda (movieID, ((predictRating, numRatings), movieName)): (predictRating, movieName)))

predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[0])
print ('My highest rated movies as predicted (for movies with more than 75 reviews):\n%s' %
        '\n'.join(map(str, predictedHighestRatedMovies)))

