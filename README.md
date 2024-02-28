YouTube Data Analysis

Overview:

This Python script conducts an in-depth analysis of a dataset containing information about various YouTube content creators. The analysis covers data exploration, trend analysis, 
audience study, performance metrics, content categories, brands and collaborations, and benchmarking.


Data Exploration:

The script begins by loading the dataset using Pandas and providing an overview of the structure with the first 5 rows and data information.



Trend Analysis:


Missing Data:

The presence of missing data is checked, and basic statistics are explored to identify outliers. Scatter plots and box plots are used to visualize outliers, 
and a function (find_outliers_IQR) is defined to detect outliers using the Interquartile Range (IQR). The script also includes a function (impute_outliers_IQR) to impute outliers using the median.


Top Streamers Analysis:

The top 10 YouTube content creators based on subscribers are identified and visualized through a count plot and a bar plot of average performance metrics.

Correlation Analysis:

The correlation between the number of subscribers and likes/comments is calculated and visualized using a heatmap.


Audience Study:


Country Analysis:

The occurrences of each country are counted, and the average number of subscribers per country is visualized through a bar plot. A heatmap is used to explore regional preferences for content categories.


Performance Metrics:

Average performance metrics (subscribers, visits, likes, comments) are calculated and visualized.



Content Categories:

The distribution of content categories is explored through a bar plot. Categories with exceptional performance metrics are identified.


Brands and Collaborations:

The script cleans the data by dropping rows with missing values in 'Visits' and 'Subscribers'. A scatter plot is created to visualize the relationship between performance metrics and brand collaborations. The correlation coefficient between visits and brand collaborations is calculated.


Benchmarking:

The script reads the dataset again and calculates average values for each performance metric. It identifies top-performing content creators based on these averages.


Conclusion:

This comprehensive analysis provides insights into the YouTube content creator landscape, including trends, audience preferences, performance metrics, and collaborations.
