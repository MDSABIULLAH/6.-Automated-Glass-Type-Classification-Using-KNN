"""

1.	Business Problem

1.1.	What is the business objective?
1.1.	Are there any constraints?

3.	Data Pre-processing
3.1 Data Cleaning, Feature Engineering, etc.


4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
4.2.	Univariate analysis.
4.3.	Bivariate analysis.

5.	Model Building
5.1	Build the model on the scaled data (try multiple options).
5.2	Perform KNN and use cross-validation techniques to get the optimum K value.
5.3	Train and test the model and perform cross-validation techniques. Compare accuracies, precision, and recall and explain them in the documentation.
5.4	Briefly explain the model output in the documentation. 
6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?


Problem Statement:

A glass manufacturing plant uses different earth elements to design new glass materials based on customer requirements. 
For that, they would like to automate the process of classification as it’s a tedious job to manually classify them. 
Help the company achieve its objective by correctly classifying the glass type based on the other features using KNN algorithm. 


"""







"""

Business Objective: Automate the classification process of glass types to improve efficiency and meet customer requirements.

Business Constraint: Minimize costs associated with the manual classification of glass types.



Success Criteria:

Business Success Criteria: Reduce the time required for identifying the type of glass by 40 to 50%.

ML Success Criteria: Achieve an accuracy of 70% in classifying glass types using the KNN algorithm.

Economic Success Criteria: Increase revenue by 5 to 10% through enhanced process efficiency and accuracy in classification

"""








"""

Data Dictionary:
    
    

Column           Data Type                    Description 
            
RI               float64                      Refractive Index of the glass. 
Na               float64                      Sodium content (weight %) in the glass. 
Mg               float64                      Magnesium content (weight %) in the glass. 
Al               float64                      Aluminum content (weight %) in the glass. 
Si               float64                      Silicon content (weight %) in the glass. 
K                float64                      Potassium content (weight %) in the glass.
Ca               float64                      Calcium content (weight %) in the glass. 
Ba               float64                      Barium content (weight %) in the glass. 
Fe               float64                      Iron content (weight %) in the glass.
Type             int64                        Type of glass (categorical) - a numerical identifier for glass categories.






"""








# Importing the necessary libraries
import pandas as pd  # For data manipulation
from sklearn.preprocessing import MinMaxScaler  # For scaling features
from sklearn.pipeline import Pipeline  # For creating a pipeline of processes
from sklearn.compose import ColumnTransformer  # For applying transforms to multiple columns
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.neighbors import KNeighborsClassifier  # To implement the KNN algorithm
from sklearn.preprocessing import LabelEncoder  # For converting class labels to integers
import sklearn.metrics as sket  # For evaluating the performance of the model
import matplotlib.pyplot as plt  # For plotting
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
import pickle  # For serializing and de-serializing Python objects
import joblib  # For saving and loading large numpy arrays and models
from sqlalchemy import create_engine, text  # Importing functions from SQLAlchemy for database connection and SQL text handling
from urllib.parse import quote              # Importing the 'quote' function from the urllib.parse module to safely encode special characters in URLs
import sweetviz as sv                       # Importing the Sweetviz library for data visualization and analysis, allowing us to create visual reports from dataframes
from feature_engine.outliers import Winsorizer  # Importing the Winsorizer class from the feature_engine library to handle outlier treatment based on Winsorization.
from sklearn.impute import SimpleImputer        # Importing the SimpleImputer class from scikit-learn for handling missing values in datasets through various strategies (mean, median, etc.).


# Reading the CSV file containing the dataset
glass = pd.read_csv("C:/Users/user/Desktop/data science assignment question/glass.csv")
glass.head()  # Displaying the first few rows of the dataset
glass.info()  # Displaying information about the DataFrame, such as data types and non-null counts





# CREATING ENGINE TO CONNECT DATABASE
user = 'root'                      # Database username
pw = '12345678'                    # Database password
db = 'univ_db'                     # Database name

# Create a connection engine to the MySQL database using pymysql
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

# SELECTING THE ENTIRE DATA FROM DATABASE
# Write the DataFrame 'glass' to the 'glass' table in the database
# If the table exists, replace it. Do not write the index as a column
glass.to_sql('glass', con=engine, if_exists='replace', index=False)

# Prepare a SQL query to select all records from the 'glass' table
sql = text('select * from glass')

# Execute the SQL query and read the result into a DataFrame
glass = pd.read_sql_query(sql, con=engine)

# Display information about the DataFrame, including data types and non-null counts
glass.info()

# Display the first few rows of the DataFrame
glass.head()




# Generating the Report.
report = sv.analyze(glass)

# Showing the Report in the Html File. 
report.show_html('Glass_Report.Html')







# Evaluating the distribution of the target variable
glass["Type"].value_counts()  # Displaying the count of each class in the 'Type' column



# Converting the 'Type' column from integer to object to treat it as categorical data
glass["Type"] = glass["Type"].astype("object")



# Checking the count again to confirm changes
glass["Type"].value_counts()
glass.head()



# Checking for missing values in the dataset
glass.isnull().sum()  # Displaying the number of missing values in each column




# Displaying the DataFrame info again after conversion to check data types
glass.info()



# Checking for duplicate rows in the dataset
glass.duplicated().sum()  # Counting the number of duplicate rows

# Removing duplicate rows
glass.drop_duplicates(inplace=True)

# Verifying duplicates removal
glass.duplicated().sum()





# Separating the input and output columns
# Input features
glass_X = glass.iloc[:, 0:9]
glass_X = pd.DataFrame(glass_X)

glass_X.head()






# # Create a box plot for each column in glass_X
# plt.figure(figsize=(50, 35))  # Set the figure size

# # Creating box plots
# plt.boxplot([glass_X[col] for col in glass_X.columns], labels=glass_X.columns)

# # Add titles and labels
# plt.title('Box Plots of All Columns in glass_X')
# plt.xticks(rotation=45)  # Rotate x-axis labels 
# plt.ylabel('Values')

# # Show the plot
# plt.tight_layout()  # Adjust layout to make room for label rotation
# plt.show()



# # Storing the Ba column so that after winsorization this column concatenated to glass_X.
# glass_X_Ba = glass['Ba']



# # Dropping the Ba column because it has very low Variation and it Doesn't fit for the iqr method.
# glass_X.drop(columns='Ba', inplace=True)




# # Step 1: Initializing the Winsorizer
# winsor = Winsorizer(capping_method='iqr',  # Choose IQR rule for boundaries
#                     tail='both',          # Cap both tails
#                     fold=1.5,             # Multiplier for IQR
#                     variables=list(glass_X.columns))  # Specify all columns to be Winsorized



# # Step 2: Fit the Winsorizer
# outlier = winsor.fit(glass_X)  # Fit to all specified columns



# # Step 3: Save the Winsorizer model
# joblib.dump(outlier, 'winsor.joblib')

# # Step 4: Transform the data to apply Winsorization
# glass_X[:] = outlier.transform(glass_X)  # Apply Winsorization to all columns

# # Step 5: Create box plots for the Winsorized data
# glass_X.plot(kind='box', subplots=True, sharey=False, figsize=(12, 8))

# # Step 6: Show the plots
# plt.suptitle('Box Plots of Winsorized Data (All Columns)')
# plt.show()




# # Combining the DateFrame and Series
# glass_X = pd.concat([glass_X, glass_X_Ba], axis=1)  # axis=1 for column-wise concatenation
# glass_X.head()


# # Reorder columns to put 'Ba' before 'Fe'
# new_order = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
# glass_X = glass_X[new_order]  # Reindexing the DataFrame
# glass_X.head()





# Output feature
glass_Y = glass.iloc[:, -1]
glass_Y = pd.DataFrame(glass_Y)






# Due to no categorical features requiring encoding and no missing values, no further preprocessing was needed.




# Creating a pipeline for scaling input features
pipeline_1 = Pipeline(steps=[
    ('missing_impute',SimpleImputer(strategy='mean')),
    ("scale", MinMaxScaler())])

# Serialization: Saving the scaling pipeline to a file using joblib
joblib.dump(pipeline_1, "pipeline_minmaxscaler.joblib")

# Checking the current working directory
import os
os.getcwd()




# Creating a column transformer to apply MinMaxScaler to all feature columns
preproccesed = ColumnTransformer([("transform1", pipeline_1, glass_X.columns)])

preproccesed_fit =preproccesed.fit(glass_X)

# Serialization: Saving the column transformer to a file
joblib.dump(preproccesed_fit, 'columntransformer_minmaxscaler.joblib')




# Transforming the feature data
glass_X_clean = pd.DataFrame(preproccesed.fit_transform(glass_X), columns=glass_X.columns)



# Converting the output column to a numpy array for compatibility with machine learning algorithms
Y_array = np.array(glass_Y["Type"])



# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(glass_X_clean, Y_array, test_size=0.2, random_state=0)

# Displaying the first few rows of the datasets to verify the split
X_train.head()
X_test.head()

# Converting the test and train sets into numpy arrays
x_train = np.array(X_train)
x_test = np.array(X_test)

# Convert target arrays to integer 
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)





# Training a KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(x_train, Y_train)

# Predicting the training set results
pred_train = knn.predict(x_train)

# Generating a confusion matrix with predictions and real values
pd.crosstab(Y_train, pred_train, rownames=["Actual"], colnames=["Predicted"])

# Calculating accuracy on the training set
print(sket.accuracy_score(Y_train, pred_train))

# Predicting the test set results
pred_test = knn.predict(x_test)

# Calculating and printing the accuracy on the test set
print(sket.accuracy_score(Y_test, pred_test))

# Generating and displaying the confusion matrix
cm = sket.confusion_matrix(Y_test, pred_test)

# Displaying the confusion matrix using ConfusionMatrixDisplay
cmplot = sket.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["1", "2", "3", "5", "6", "7"])
cmplot.plot()  # Plotting the confusion matrix

# Setting the title and labels for the confusion matrix plot
cmplot.ax_.set(title="Glass Type Classification", 
               xlabel="Predicted Value", ylabel="Actual Value")








# Exploring how accuracy changes with different values of k in KNN
accuracy = []

# Iterate over a range of odd numbers for k
for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh = neigh.fit(x_train, Y_train)  # Fit the model
    train_accuracy = np.mean(neigh.predict(x_train) == Y_train)  # Calculate train accuracy
    test_accuracy = np.mean(neigh.predict(x_test) == Y_test)  # Calculate test accuracy
    # Append the difference and both accuracies to the list
    accuracy.append([train_accuracy - test_accuracy, train_accuracy, test_accuracy])

# Plotting the training and test accuracies
plt.plot(np.arange(3, 50, 2), [i[1] for i in accuracy], "ro-", label="Train Accuracy")
plt.plot(np.arange(3, 50, 2), [i[2] for i in accuracy], "bo-", label="Test Accuracy")
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.title("K-NN Varying Number of Neighbors")
plt.legend()
plt.show()







# Hyperparameter tuning using GridSearchCV
# Defining a range of k values to test
rang = list(range(3, 50, 2))
param_grid = dict(n_neighbors=rang)

# Initializing GridSearchCV to find the best k
grid = GridSearchCV(knn, param_grid, cv=5, return_train_score=False, verbose=1)

# Fit the grid to the training data
KNN_new = grid.fit(x_train, Y_train)

# Output the best number of neighbors found
print(KNN_new.best_params_)

# Calculate and print the best cross-validation score
accc = (KNN_new.best_score_ * 100)
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accc))

# Making predictions on the test set with the tuned model
pred_3 = grid.predict(x_test)
print(pred_3)


# Calculating the test accuracy with the best model
print(sket.accuracy_score(Y_test, pred_3))




# Plot and display the confusion matrix for the best model predictions
cm = sket.confusion_matrix(Y_test, pred_3)
cmplot = sket.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2', '3', '5', '6', '7'])
cmplot.plot()
cmplot.ax_.set(title='Glass Type Classification - Confusion Matrix',
               xlabel='Predicted Value', ylabel='Actual Value')







# Saving the best model from grid search results
knn_best = KNN_new.best_estimator_
print(knn_best)

# Serialize the best model using pickle
pickle.dump(knn_best, open('knn.pkl', 'wb'))

# Check current working directory
import os
os.getcwd()







# Briefly explain the model output in the documentation. 

'''

Model Output Explanation
The model used is a K-Nearest Neighbors (KNN) classifier which predicts the type of glass based on input features. 
After hyperparameter tuning and evaluating with cross-validation, the best model was selected to make predictions on the test data.

Actual vs. Predicted Values:
The Y_test array contains the actual labels for the glass types.
The pred_3 array contains the predictions made by the KNN classifier on the test dataset.
Examining these arrays shows which predictions (from pred_3) match with the actual values (Y_test).

Sample Predictions:

The classifier successfully predicted certain glass types, evident from matches between Y_test and pred_3 (e.g., 7, 1, 2, 5).
However, there are differences where the model incorrectly predicted the glass type 
(e.g., at index 4 where the actual is 3 but predicted as 1, at index 12 where the actual is 3 but predicted as 2).



Confusion Matrix Analysis
To further analyze the model's output, the confusion matrix provides a detailed look at how often types are correctly predicted 
versus how often they are misclassified.

True Positives: Diagonal values in the confusion matrix, indicating correct predictions.
False Positives/Negatives: Off-diagonal values, showing instances where predictions differ from actual values.


'''





# Benefits and Impact of the Solution


"""
Increased Efficiency:

Automation: The manual classification of glass types is labor-intensive and prone to human error. 
Automating this process significantly speeds up operations, allowing employees to focus on more complex tasks that require human intervention.
Consistency and Accuracy: The use of a machine learning model ensures consistent decision-making, reducing variability and improving 
the reliability of classification outcomes.



Cost Reduction:

Operational Savings: Automating the classification process reduces the need for extensive manual labor, 
potentially leading to decreased labor costs and associated overheads.
Error Reduction: Minimizing classification errors reduces costs associated with rework or material waste.
Scalability:

Handling Varying Loads: The model is capable of handling varying amounts of data, supporting the company in 
times of peak demand without requiring additional staffing.
Adaptability: As new glass compositions are developed, the model can be easily retrained or updated to 
incorporate new data and maintain performance.



Improved Customer Satisfaction:

Faster Turnaround: Quicker classification processes mean faster fulfillment of customer orders,
potentially improving customer satisfaction and retention.
Custom Solutions: Accurate classification supports the development of custom glass solutions tailored to specific customer needs,
enhancing the value proposition offered by the company.


Data-Driven Insights:

Strategic Decision-Making: Analyzing classification data can provide insights into production trends, customer demands, 
and material usage, informing strategic business decisions.
Quality Control: Continuous monitoring and analysis help maintain high quality and consistency in glass products.







