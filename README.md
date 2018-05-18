# CS230

___Description___: This NN uses social media data to predict price change among cryptocurrencies. The goal of the algorithm is to predict positive change with high accuracy, and predicts over the below-mentioned time periods.

First, the algorithm aggregates all the data from every currency to train a general model with very low variance. This model is used to initialize custom models for each currency. Each currency is then trained on it's own unique dataset. This approach takes advantage of the massive amount of data availible across all currencies for better training, yet still mantains the advantage of customization per currency. It also helps to avoid overfitting.

Additionally, once the primary model is trained, the customized models will be trivial to execute.

Some unique facets of this algorithm include:
_Data Clensing_: Instead of just mining tweets that contain mentions of the currency we're looking for, we're going to parse the tweets for spam and promotional tweets. Tweets with the objective of promoting currencies should not be analyzed. To clense the data in this way, we will use prebuilt algorithms by other engineers.

_Sentiment Analysis_: All tweets will be analyzed with a standard sentiment analysis, and their relative importance will be multiplied by their sentiment.

_Weighted Importance_: Tweets will be weighted by how popular they are, as a metric of retweets and favorites.

_Modified Time Data_: All the data will be mined per minute. We will then use weighted averages over time period m, to transform this data into a set of data. This will allow the algorithm to find the relationship between the last hour's tweets, and the change in the following minute. This creates a much more dynamic set of data. The same thing will be done with output, and the weighted average time period will simply be changed to predict for different time periods.

This modification is extremely important as it allows for significantly more training & label data. For example, a normal algorithm to predict a days price change of a specific currency will have 1000 data points relating to 1000 days. For those same 1000 days, my algorithm, has 144 million (1000*24*60*100) data points.

_Depth_: The aforementioned modefication of the data allows for a vastly expanded network. With millions of pieces of data the main training model can be much deeper. This allows for an examination of relationships previously uncapturable by Neural Networks. Though a very deep network is not in itsef a unique feature, the depth of this network in relation to the type of data is novel.

_Unique Loss Function_: I do not use a traditional loss function. The goal of this algorithm is to predict positive change. I do not care if the price of a currency is going to drop, because I cannot make money off a currency dropping. I do however care about the accuracy in predicting positive change. When the algorithm predicts positive change, it should be very confident. We can achieve this by creating a loss function that increases penalties when the actual change is negative. I created my own loss function that penalizes wrong negative choices greatly, but gives lower loss values to accurate positive predictions. This trains the model to predict positive change effectively.

The  loss function is as follows:
X = Prediction
Y = Actual
L(X, Y) = (Y - X)^2 • e^(sigmoid(Y)•x - Y)

_Test Metric_: To test the algorithm's performance, negative predictions do not matter as they will be ignored in trading. Additionally, the only predictions that matter are those which are most confident. If my algorithms makes 500 predictions, there only needs to be one or two great predictions. Therefore, the test metric is two-fold, and designed to optmize for best-case predictions.

The first test metric, for the general model, is calculating the loss solely on positive predictions.

The second test metric, is on each currency's customized model, to test what percent of positive predictions are within one standard deviation of the actual.

##Data
###Sources
A) Aggregated&Sentimented Tweet data
B) Aggregated&Sentimented Tweet data only for top crypto influencers
C) Aggregated&Sentimented Reddit data
D) Aggregated&Sentimented Reddit data for top influencers
E) Aggregated&Sentimented News data
F) Aggregated&Sentimented News data on top publications
G) Previous price behavior
H) Trade volume

All data is per minute
###Preprocessing


##Initialization

##Training

##Time Periods
1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour, 2 hours, 4 hours, 6 hours, 12 hours, 24 hours, 2 days, 7 days

##Disadvantages to this approach
Because this approach optimizes for positive predictions and optomizes for false negatives, not false positives, it will likely miss some bull periods in the market.

