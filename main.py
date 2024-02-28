# 1 - Data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv(r"C:\Users\Surav\Desktop\youtubers_df.csv")

print("Entire DataFrame:")
print(data)

# Exploring the dataset structure
print('Exploring the first 5 rows:')
print(data.head())
print(data.info())
###############################################################################################

# 2 - Trend Analysis
# Check for missing data
print('Missing Data:')
print(data.isnull().sum())

# Explore basic statistics to identify outliers
# print(data.describe())
var = data.describe()[['Suscribers', 'Visits', 'Likes', 'Comments']]
print("Statistics",var)

# visualize the outliers here #
# Scatter Plot
plt.scatter(data['Suscribers'], data['Visits'])
plt.xlabel('Suscribers')
plt.ylabel('Visits')
plt.title('Scatter Plot of Subscribers vs. Visits')
plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
plt.boxplot(data['Suscribers'])
plt.ylabel('Suscribers')
plt.title('Box Plot of Subscribers')
plt.show()

# Exploring the outliers using IQR
print('Finding outliers using IQR')
def find_outliers_IQR(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    outliers = data[((data < (q1 - 1.5 * IQR)) | (data > (q3 + 1.5 * IQR)))]
    return outliers


outliers = find_outliers_IQR(data[["Suscribers", "Visits", "Likes", "Comments"]])
print("Number of Outliers: " + str(len(outliers)))
print("Max Outlier value: " + str(outliers.max()))
print("Min Outlier value: " + str(outliers.min()))
print(outliers)

def impute_outliers_IQR(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    IQR = q3 - q1
    upper = q3 + 1.5 * IQR
    lower = q1 - 1.5 * IQR
    data = np.where((data > upper) | (data < lower), data.median(), data)
    return data

##############################################################################################

# 2 - Trend Analysis

# Count the occurrences of each category
print('Identifying trends among the top youtube streamers')
data["Categories"].value_counts()

top_streamers = data.sort_values(by='Suscribers', ascending=False).head(10)

# Category Analysis
plt.figure(figsize=(12, 6))
sns.countplot(x='Categories', data=top_streamers)
plt.title('Top Streamers by Category')
plt.xticks(rotation=45, ha='right')
plt.show()

# Performance Metrics Visualization
average_metrics = top_streamers[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()
average_metrics.plot(kind='bar', figsize=(10, 6), rot=0)
plt.title('Average Performance Metrics for Top Streamers')
plt.xlabel('Metrics')
plt.ylabel('Average Value')
plt.show()


# Correlation between the number of sucscribers and the number of likes and comments
correl = data['Suscribers'].corr(data['Likes'])
print("Correlation between Number of Suscribers and Likes:\n", correl)

correl = data['Suscribers'].corr(data['Comments'])
print("Correlation between Number of Suscribers and Comments:\n", correl)

# Correlation matrix
correlation_matrix = data[['Suscribers', 'Visits', 'Likes', 'Comments']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

########################################################################################################

# 3 - Audience Study

# Count the occurrences of each country
country_counts = data['Country'].value_counts()

# Group by Country and calculate mean, median, or other relevant statistics
country_stats = data.groupby('Country')['Suscribers'].mean()

# Display the statistics for audience distribution by country
print(country_stats)

# Visualize the distribution of audiences by country
plt.figure(figsize=(12, 8))
country_stats.plot(kind='bar')
plt.title('Audience Distribution by Country')
plt.xlabel('Country')
plt.ylabel('Average Audience')
plt.xticks(rotation=45)

plt.show()

region_category_stats = data.groupby(['Country', 'Categories'])['Suscribers'].mean().unstack()
# Display the statistics for audience distribution by country and category
print(region_category_stats)

# Heatmap for regional preferences
plt.figure(figsize=(12, 8))
sns.heatmap(region_category_stats, cmap='viridis', annot=True, fmt=".0f", linewidths=.5)
plt.title('Regional Preferences for Content Categories')
plt.xlabel('Categories')
plt.ylabel('Country')
plt.show()

##############################################################################################

# 4 - Performance Metrics

# Calculate average metrics
average_metrics = data[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

# Display average metrics
print("Average Subscribers:", average_metrics['Suscribers'])
print("Average Visits:", average_metrics['Visits'])
print("Average Likes:", average_metrics['Likes'])
print("Average Comments:", average_metrics['Comments'])


# Visualize average metrics
plt.figure(figsize=(12, 6))
sns.barplot(x=average_metrics.index, y=average_metrics.values)
plt.title('Average Performance Metrics')
plt.ylabel('Mean Count')
plt.show()


##############################################################################################

# 5 - Content Categories
# Explore the distribution of content categories
category_dist = data['Categories'].value_counts()

# Display the number of streamers per category
print("Number of streamers per category:")
print(category_dist)

# Plot the distribution of content categories
plt.figure(figsize=(12, 6))
sns.barplot(x=category_dist.index, y=category_dist.values)
plt.title('Distribution of Content Categories')
plt.xlabel('Category')
plt.ylabel('Number of Streamers')
plt.xticks(rotation=45)
plt.show()

# Identify categories with exceptional performance metrics
# You can choose metrics like average likes, comments, etc.
performance_metrics = ['Likes', 'Comments', 'Visits']

for metric in performance_metrics:
    # Calculate average metric per category
    avg_metric_per_category = data.groupby('Categories')[metric].mean()

    threshold_value = 16000

    # Identify categories with exceptional performance (e.g., above a certain threshold)
    exceptional_categories = avg_metric_per_category[avg_metric_per_category > threshold_value]

    # Display the results
    print(f"\nCategories with exceptional {metric} performance:")
    print(exceptional_categories.sort_values(ascending=False))

#################################################################################################

# 6 - Brands and Collaborations
data_cleaned = data.dropna(subset=['Visits', 'Suscribers'])
print(data_cleaned[['Visits', 'Suscribers']].dtypes)
data_cleaned['Visits'] = pd.to_numeric(data_cleaned['Visits'], errors='coerce')
data_cleaned['Suscribers'] = pd.to_numeric(data_cleaned['Suscribers'], errors='coerce')
# Scatter plot of performance metrics vs. brand collaborations
plt.figure(figsize=(12, 8))
plt.scatter(data_cleaned['Visits'], data_cleaned['Suscribers'], c='blue', alpha=0.5)
plt.title('Performance Metrics vs. Brand Collaborations')
plt.xlabel('Visits')
plt.ylabel('Brand Collaborations')
plt.show()

# Calculate correlation coefficients
correlation_visits_links = data_cleaned['Visits'].corr(data_cleaned['Suscribers'])
print(f"Correlation between Visits and Brand Collaborations: {correlation_visits_links}")

############################################################################################################

# 7 - Benchmarking
df = pd.read_csv(r"C:\Users\Surav\Desktop\youtubers_df.csv")


# Calculate average values for each performance metric
average_subscribers = df['Suscribers'].mean()
average_visits = df['Visits'].mean()
average_likes = df['Likes'].mean()
average_comments = df['Comments'].mean()

# Identify top-performing content creators
top_performers = df[
    (df['Suscribers'] > average_subscribers) &
    (df['Visits'] > average_visits) &
    (df['Likes'] > average_likes) &
    (df['Comments'] > average_comments)
]

# Display the top-performing content creators
print("Top-performing content creators:")
print(top_performers[['Rank', 'Username', 'Categories', 'Suscribers', 'Visits', 'Likes', 'Comments', 'Links']])







