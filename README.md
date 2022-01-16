# CA_ONE_Machine_Learning
 
2 | P a g e
Table of Contents 
Intuition ................................................................................................................................................... 3 
Dataset source ......................................................................................................................................... 3 
SECTION 1: Choice of dependent and independent variables and selection of algorithm .................... 3 
Choice of dependent and independent variables ................................................................................. 3 
Variable ........................................................................................................................................... 3 
Independent variable ....................................................................................................................... 3 
Dependent variable ......................................................................................................................... 3 
Selection of Algorithm ........................................................................................................................ 4 
Linear Regression ........................................................................................................................... 4 
SECTION 2: Data Preparation ................................................................................................................ 5 
Data pre-processing ............................................................................................................................ 7 
SECTION 3: Feature Selection ............................................................................................................. 17 
Variance inflation factor (VIF) ......................................................................................................... 17 
Mathematical rule ......................................................................................................................... 17 
Type conversion ............................................................................................................................ 17 
P – value ............................................................................................................................................ 20 
Mathematical rule ......................................................................................................................... 20 
SECTION 4: Model Development and Evaluation ............................................................................... 22 
Model development .......................................................................................................................... 22 
Prediction ...................................................................................................................................... 22 
Evaluation ......................................................................................................................................... 23 
SECTION 5: Model Comparison .......................................................................................................... 24 
Regularization ................................................................................................................................... 24 
Lasso ............................................................................................................................................. 24 
Ridge Regression .......................................................................................................................... 27 
ElasticNet ...................................................................................................................................... 28 
Optimization Algorithm .................................................................................................................... 28 
Stochastic Gradient Descent (SGD) .............................................................................................. 28 
Conclusion ............................................................................................................................................ 29 
References ............................................................................................................................................. 30 
3 | P a g e
Intuition 
Understanding the dataset description, we are attempting to determine when bikes are more 
commonly used based on temperature, humidity, season, month, year, weekdays, holidays, 
windspeed, and weather conditions. 
Dataset source 
https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
SECTION 1: Choice of dependent and independent variables and 
selection of algorithm 
Choice of dependent and independent variables 
Variable 
A variable is a property of an object. A variable can be independent or dependent on another variable. 
Independent variable 
A variable that is unaffected by other variables, it also has no relationship with other variables. For 
example, except for the cnt column, all the variables in the dataset we worked on are independent 
variables because they do not rely on other features to determine their value. 
Dependent variable 
A variable that is influenced by other factors. It also has a relationship with other variables. Label 
column is another name for the dependent variable. The cnt column, for instance, is our dependent 
variable because it is affected by independent variables such as season, yr, mnth, holiday, weekday, 
workingday, weathersit, temp, atemp, hum, windspeed, casual, and registered.
 
4 | P a g e
Selection of Algorithm 
Here we are applying linear regression to our model. Before we apply linear regression to our model, 
there are some rules we must follow. 
Linear Regression 
(Tibshirani et al., 2021). Simple linear regression is a very simple method for predicting a quantitative 
response Y from a single predictor variable X. It is presumptively assumed that X and Y have a linear 
relationship. Linear regression is a method for determining the best straight line fit to the provided 
data, i.e., the best linear correlation between the independent and dependent variables. 
For univariate, 
𝑦 = 𝑚𝑥 + 𝑐
For multivariate, 
𝑦 = 𝛽଴𝑥଴ + 𝛽ଵ𝑥ଵ + 𝐶
Where y = Dependent variable 
 m = Slope of the line 
 x = Independent variable 
 c = Intercept of the line 
 C = Intercept of the line 
 𝛽଴ , 𝛽ଵ = slope of 𝑥଴ and 𝑥ଵ
 𝑥଴ , 𝑥ଵ = Independent variable which are contributing to predicting the y value 
Assumptions 
 X (independent variables) should be correlated with Y (dependent variable). 
 The residuals mean should be zero. 
 Error terms are not allowed to be correlated to one another. 
 X residuals must be uncorrelated. 
 (Codecademy, 2008). The variance of the error term must be constant. 
 No multicollinearity i.e., no relationships with the features themselves. 
 Error terms are supposed to be normally distributed when plotted on graph. 
If and only if these assumptions apply, we could use a linear regression algorithm. 
5 | P a g e
SECTION 2: Data Preparation 
Data preparation comprises both information gathering and data cleansing. We collect data from a 
variety of sources in data gathering. Cleaning data involves removing null and missing values. 
Importing all the required libraries. Numpy is used for numerical operations, while pandas are used 
for creating, reading, updating, and deleting objects. The data is visualized using Matplotlib, seaborn, 
and ProfileReport. Standardizing and normalizing are accomplished using StandardScalar and 
MinMaxScalar, respectively. We utilized Lasso, Ridge, and ElasticNet for regularization. 
LinearRegression is utilized because we choose this approach to apply to our model, and 
train_test_split is used to separate the data into train and test data. 
Using the pandas function read_csv, read data from day.csv file. With the exception of the dteday 
column, we can see that most of the column values are numerical. 
Checking the dataset's dimensions. 
6 | P a g e
There are no missing values, as evidenced by the count of each column. 
examining the total number of null values in each column. 
7 | P a g e
Data pre-processing 
Before training the model, we must ensure that there are no missing, null, or category data for our 
linear regression model, as our algorithm requires numerical values to predict the outcome. This data 
pre-processing phase is required since it increases the learning and accuracy of our model. 
The dteday column is transformed to datetime type in this case because we are attempting to break 
this one column into two independent columns such as year and month. 
8 | P a g e
We removed the yr and mnth columns since we extracted the year and month from the dateday 
column, which is more accurate. 
The additional variable holiday is being dropped because the workingday section has all the necessary 
data. Dropping the dteday column because we already have a year and a month, as well as the fact that 
we cannot operate with non-numerical columns and the instant column is a non-essential column. 
To eliminate confusion, the columns have been renamed. 
9 | P a g e
According to the dataset description, the ordinal and nominal columns are being converted into 
categorical columns and transforming year into numerical column. 
For data visualization, we attempted to limit the number of lines. When we use ProfileReport to 
visualize our dataset, we can see the columns overview, variables, interactions, correlations, missing 
values, and sample. In Overview, we have the number of columns, the number of rows, the number of 
missing values, the number of duplicate rows, the number of categorical columns, and the number of 
numerical columns. Each individual column statistical data is available under the variables tab. 
The graphs between two columns and their relationships can be seen in the interaction tab. 
10 | P a g e
 
11 | P a g e
In correlations tab, we can see whether there is a positive or negative correlation between each 
variable or column; there are several correlations like as Persons, Spearman's, and so on. We 
attempted to demonstrate some. 
 
12 | P a g e
To find out if there is a missing value, go to the missing tab. Fortunately, there are no missing values. 
 
13 | P a g e
The top ten columns and bottom ten columns of the dataset were visible under the sample tab. 
As mentioned in assumptions of linear regression, trying to avoid multicollinearity for the 
independent variables. Checking the plots of temp and atemp using ProfileReport from pandas 
profiling, you can see that there is a substantial correlation between them, as well as casual and 
registered. As a result, the columns listed below have been removed. 
14 | P a g e
15 | P a g e
There are still categorical columns in the preceding example, and we need to do something about 
them because our model can forecast data based on numerical values. Using the pandas function get 
dummies, convert these categorical columns to numbers. This function adds 1 everywhere the value 
is. For example, in the month column, wherever Jan is indicated as 1 and where it is not designated as 
0. Apply to the remaining categorical columns as well, such as month, season, weekday, workingday, 
and weathersit. 
Concatenating the columns generated above to the DataFrame. As per the DataFrame, we have a total 
of 33 columns. 
Remove all categorical because it has already been transformed and we no longer need it, leaving us 
with 28 columns. 
Temp, humidity, and windspeed have values between 0 and 1, whereas the rest of the columns have 
values either 0 or 1, except for the count column, which has values that are far apart from 0 and 1. 
16 | P a g e
Attempting to bring all columns other than dummy columns under one scale, using normalization to 
do so. The count field spans between 0 and 1 after normalization. Temp, humidity, and windspeed 
have previously been normalized in accordance with the dataset description. 
After identifying the independent and dependent variables, extract them from the Dataframe to 
prepare them for feature selection from the independent variables. 
17 | P a g e
SECTION 3: Feature Selection 
There are two approaches for selecting a feature. The first is the Variance inflation factor, while the 
second is by p-values. 
Variance inflation factor (VIF) 
(Tibshirani et al., 2021). The variance inflation factor quantifies the degree to which the behavior 
(variance) of an independent variable is affected or inflated by its interaction/correlation with other 
independent variables. The variance inflation factor provides a quick estimate of how much a variable 
affects the standard error of the regression. 
Mathematical rule 
There is only one criterion for VIF: if the VIF for any column exceeds 10 (VIF > 10), we simply 
delete the column because it has a higher correlation/contribution to error terms. 
Type conversion 
As Y variable has an array type, we change X to an array type as well, because the VIF function 
accepts array type values. 
The variance inflation component is being imported from the relevant library. Here, "arr" has all 
features and "i" contains columns, and these parameters are passed to the variance inflation factor 
function. The values of vif for each column are listed in the vif column, while the names of the 
features are listed in the "feature" column. 
 
18 | P a g e
Some of the characteristics in the preceding DataFrame have values larger than 10, i.e., VIF > 10. By 
removing columns like temp, humidity, Working_day, and summer, the remaining features are 
involved in forecasting the target variable. This is how we decide which features to include. Don't be 
confused by the dropping of the count column; our initial DataFrame after Data cleaning still retains 
all its properties, including our target variable, because we didn't use the inplace option to 
permanently eliminate it. 
19 | P a g e
After omitting some features, double-check the VIF values to ensure that no features are contributing 
to error terms, as this is one of our linear regression assumptions. 
We can proceed with model training because there are no VIFs greater than 10. 
 
20 | P a g e
P – value 
A p-value is a statistic that expresses the likelihood that an observed difference may have occurred by 
chance. The statistical significance of the observed difference is increased by lowering the p-value. 
The Ordinal Least Square method (OLS) can be used to determine the P value. This function has two 
essential parameters: formula and DataFrame. 
Mathematical rule 
If p < 0.05, the column is significant, which means that more than 95% of the rows in the column are 
contributing. Otherwise, if p > 0.05, the column has no significance, and we remove all ones that meet 
this criterion. 
21 | P a g e
Remove columns with p > 0.05 since they don't contribute much and have little relevance, such as 
Aug, Feb, Jan, July, June, Mar, May, Oct, Mon, Sat, Thu, Tue, and Wed, and deep copying the initial 
DataFrame so that it may be used for model comparison and evaluation ensuring the original 
DataFrame remains unchanged. 
These are the features we chose to evaluate our model. Pass the target variable and the rest of the 
features to the formula parameter. 
 
22 | P a g e
SECTION 4: Model Development and Evaluation 
Model development 
In model development, the single most critical thing you can do to properly evaluate your model is to 
avoid training it on the whole dataset. Do not train the model on the complete dataset. 
Divide the dataset into training and testing regions. In this case, train test split takes four primary 
factors into account: features, labels, test size = 0.20, which means that 20% of the data is testing data 
and the remaining 80% is training data, and random state is utilized to provide same results every 
time. 
Calling the LinearRegression constructor to create an object and attempting to train the model with 
the fit () function. 
Prediction 
Testing to see if our model is accurate enough. The predicted value is 0.101, but the actual value is 
0.110 and there isn't much of a difference. 
Using test data to validate our model and determine its performance score. 
23 | P a g e
Evaluation 
Using R square/Adjusted R square metrics, we found that our model was 75.28 % accurate by using 
linear regression. 
We obtained an adjusted R square score of 83.68 % accuracy by applying the ordinal least squares 
(OLS) approach. 
 
24 | P a g e
SECTION 5: Model Comparison 
To increase the performance of our model, one can use either regularization or an optimization 
approach in Model Comparison. 
Regularization 
This is a type of regression in which the coefficient estimates are constrained/regularized or shrunk 
towards zero. All in all, to keep away from overfitting, this technique debilitates learning a more 
convoluted or adaptable model. (Aurélien Géron, 2019b). Regularization of a linear model is often 
accomplished by restricting the model's weights. We'll now look at Ridge Regression, Lasso 
Regression, and Elastic Net, which use three distinct methods to limit the weights. This strategy is 
used to ensure that our model is consistently accurate. 
Lasso 
Regularization techniques such as Lasso regression are used. For more accurate prediction, it is 
preferred over regression approaches. Shrinkage is utilized in this model. Shrinkage is the process of 
reducing data values to a single central point known as the mean. Simple, sparse models are supported 
by the lasso approach (i.e., models with fewer parameters). This type of regression is ideal for models 
with high degrees of multicollinearity or for automating certain aspects of model selection, such as 
variable selection/parameter removal. 
Taking a look at the dataset's top 5 rows. The graphs below show that temperature and count are 
highly correlated. As previously mentioned in the context of Pandas profiling, use the tabs to navigate 
around the graphs and statistical data. 
25 | P a g e
 
26 | P a g e
The feature selection is done, and we simply need to save the features and labels in some variables. 
Train and test datasets are prepared. 
LassoCV is used to calculate the alpha value for Lasso Regression. The LassoCV class accepts four 
major arguments: alpha, which is nothing more than a shrinkage factor, cv, which indicates the 
number of cross validations to perform, max_iter, which is the maximum number of iterations to 
conduct, and normalize, which is the application of normalization. 
27 | P a g e
Now we take this shrinkage factor and feed it into Lasso Regression. In order to train the model using 
the fit () method, this is known as hyper parameter tuning. As a result, our model has the smallest 
slope possible. 
Lasso regression outperforms linear regression in terms of accuracy, with linear being 75.28 percent 
accurate and lasso being 78.50 percent accurate. 
Ridge Regression 
Ridge regression is a model tuning strategy that can be used to interpret data with multicollinearity. 
L2 regularization is achieved using this method. When there is a problem with multicollinearity, leastsquares are unbiased, and variances are big, the projected values are far from the actual values. 
Because it uses absolute coefficient values for normalization, Lasso Regression differs from ridge 
regression. Attempting to select a random alpha value between 0 and 10. 
RidgeCV use the same parameters as LassoCV. 
Our alpha is close to zero, so we'll provide it to the Ridge class and use the fit () method to train the 
model. Our score is nearly identical to that of the Lasso Regression, which is 78.70% accurate. 
 
28 | P a g e
ElasticNet 
ElasticNet is created by combining Lasso and Ridge. After determining the alpha value for the 
ElasticNet, get the l1 ratio and pass these arguments to the ElasticNet contructor. Now, train the 
model with the training dataset and calculate the R square for the ElasticNet , which is our model's 
score; our model has an accuracy of 78.63 %. 
We may conclude that our model is consistent based on the three regularization methods because all 
three models produced nearly identical scores. 
Optimization Algorithm 
(Andriy Burkov, 2019). Gradient descent is sensitive to the learning rate used. It is also sluggish when 
dealing with massive datasets. Fortunately, some important enhancements to this technique have been 
developed. Minibatch stochastic gradient descent (minibatch SGD) is a variant of the technique that 
accelerates computation by approximating the gradient using fewer batches (subsets) of training data. 
Stochastic Gradient Descent (SGD) 
Gradient Descent has been extended to include stochastic gradient descent. When fresh data is 
received, any Machine Learning function works with the same goal function to reduce error and 
generalize. To circumvent the difficulties in Gradient Descent, we choose a tiny number of samples, 
especially at each stage of the algorithm, we can sample a minibatch selected evenly from the training 
set. The minibatch size is usually set to be a modest number of instances, ranging from one to a few 
hundred. 
The learning rate is an important parameter for SGD; it is required to lower the learning rate with 
time. Eta denotes our learning rate, and the L1 or L2 ratio for the optimization method, including the 
number of cross validations, is passed here. 
29 | P a g e
Our model achieves the best score of 81.64 % when eta = 0.1 and maximum iterations = 10000. 
Conclusion 
In conclusion, all the above characteristics contribute/highly correlate in predicting the label "count," 
and individuals ride bikes more frequently on working days and on Sundays when certain parameters 
such as temperature and humidity are considered. 
 
30 | P a g e
References 
Aurélien Géron (2019). Hands-on machine learning with Scikit-Learn and TensorFlow concepts, 
tools, and techniques to build intelligent systems. O’Reilly Media, Inc. 
Andriy Burkov (2019). The hundred-page machine learning book. Quebec, Canada] Andriy Burkov. 
Tibshirani, R., Hastie, T., Witten, D. and James, G. (2021). An Introduction to Statistical Learning: 
With Applications in R. Springer. 
Codecademy. (2008). Learn to code - for free | Codecademy. [online] Available at: 
https://www.codecademy.com/. 
