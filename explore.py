import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset into a DataFrame
df = pd.read_csv('supermarket_sales.csv')

# Uncomment to check data types and initial data overview
# print(df.dtypes)  # Print the data types in the df
# print(df.head())  # Display the first few rows of the dataset
# print(df.info())  # Get a concise summary of the DataFrame
# print(df.describe())  # Get descriptive statistics for numeric columns

# Data Cleaning
# Check for missing values
# print(df.isnull().sum())  # There are no null values
# Handle missing values if any (not needed here)
# df.dropna(inplace=True)
# Check for duplicates
# print(df.duplicated().sum()) # There are no duplicates

# Data Visualization
# Uncomment to visualize the distribution of a specific column
# sns.histplot(df['column_name'])
# plt.show()

# Visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, fmt=".2f")
# plt.show()

# Select only numeric columns for analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Uncomment to calculate and visualize the correlation matrix
# correlation_matrix = numeric_df.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
# plt.title('Correlation Matrix Heatmap')
# plt.savefig('correlation_matrix_heatmap.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Uncomment to visualize distributions of numeric columns
# for column in numeric_df.columns:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(numeric_df[column], kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.savefig(f'distribution_{column}.png', format='png', dpi=300, bbox_inches='tight')
#     plt.show()

# Create a directory for boxplot PNGs if it doesn't exist
output_directory = 'boxplot_pngs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Uncomment to visualize Box Plots and save as PNG
# for column in numeric_df.columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(x=numeric_df[column])
#     plt.title(f'Box Plot of {column}')
#     plt.savefig(os.path.join(output_directory, f'boxplot_{column}.png'), format='png', dpi=300, bbox_inches='tight')
#     plt.show()

# Uncomment to save a Pairplot of numeric columns
# pairplot = sns.pairplot(numeric_df)
# pairplot.savefig(os.path.join(output_directory, 'pairplot.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Uncomment to create and save a regression plot
# plt.figure(figsize=(10, 6))
# regplot = sns.regplot(x='Rating', y='gross income', data=numeric_df)
# plt.title('Regression Plot of Rating vs Gross Income')
# plt.savefig(os.path.join(output_directory, 'regplot_rating_vs_gross_income.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Uncomment to analyze monthly sales
# df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
# df['Month'] = df['Date'].dt.month  # Extract month from Date
# monthly_sales = df.groupby('Month')['Total'].sum()  # Group by month and sum Total sales
# monthly_sales.plot(kind='bar')  # Plot monthly sales
# plt.title('Monthly Sales')
# plt.xlabel('Month')
# plt.ylabel('Total Sales')
# plt.savefig('monthly_sales.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Create a directory for saving plots if it doesn't exist
output_directory = 'plots'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Uncomment to create and save various plots
# Scatter Plot
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Rating', y='cogs', data=numeric_df)
# plt.title('Scatter Plot of Rating vs COGS', fontsize=15)
# plt.xlabel('Rating')
# plt.ylabel('COGS')
# plt.savefig(os.path.join(output_directory, 'scatter_rating_vs_cogs.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Joint Plot
# joint_plot = sns .jointplot(x='Rating', y='Total', data=numeric_df)
# joint_plot.fig.suptitle('Joint Plot of Rating vs Total', fontsize=15)
# joint_plot.savefig(os.path.join(output_directory, 'joint_rating_vs_total.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Cat Plot
# cat_plot = sns.catplot(x='Rating', y='cogs', data=numeric_df)
# cat_plot.fig.suptitle('Cat Plot of Rating vs COGS', fontsize=15)
# cat_plot.savefig(os.path.join(output_directory, 'cat_rating_vs_cogs.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# LM Plot
# lm_plot = sns.lmplot(x='Rating', y='cogs', data=numeric_df)
# lm_plot.fig.suptitle('LM Plot of Rating vs COGS', fontsize=15)
# lm_plot.savefig(os.path.join(output_directory, 'lm_rating_vs_cogs.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# KDE Plot
# plt.figure(figsize=(8, 6))
# sns.kdeplot(x='Rating', y='Unit price', data=numeric_df)
# plt.title('KDE Plot of Rating vs Unit Price', fontsize=15)
# plt.xlabel('Rating')
# plt.ylabel('Unit Price')
# plt.savefig(os.path.join(output_directory, 'kde_rating_vs_unit_price.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Line Plot
# plt.figure(figsize=(8, 6))
# sns.lineplot(x='Rating', y='Unit price', data=numeric_df)
# plt.title('Line Plot of Rating vs Unit Price', fontsize=15)
# plt.xlabel('Rating')
# plt.ylabel('Unit Price')
# plt.savefig(os.path.join(output_directory, 'line_rating_vs_unit_price.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Bar Plot for Rating vs Unit Price
# plt.figure(figsize=(5, 5))
# sns.barplot(x="Rating", y="Unit price", data=numeric_df[170:180])
# plt.title("Bar Plot of Rating vs Unit Price", fontsize=15)
# plt.xlabel("Rating")
# plt.ylabel("Unit Price")
# plt.savefig(os.path.join(output_directory, 'bar_rating_vs_unit_price.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Bar Plot for Rating vs Gender
# plt.figure(figsize=(5, 5))
# sns.barplot(x="Rating", y="Gender", data=numeric_df[170:180])
# plt.title("Bar Plot of Rating vs Gender", fontsize=15)
# plt.xlabel("Rating")
# plt.ylabel("Gender")
# plt.savefig(os.path.join(output_directory, 'bar_rating_vs_gender.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Bar Plot for Rating vs Quantity
# plt.figure(figsize=(5, 5))
# sns.barplot(x="Rating", y="Quantity", data=numeric_df[170:180])
# plt.title("Bar Plot of Rating vs Quantity", fontsize=15)
# plt.xlabel("Rating")
# plt.ylabel("Quantity")
# plt.savefig(os.path.join(output_directory, 'bar_rating_vs_quantity.png'), format='png', dpi=300, bbox_inches='tight')
# plt.show()

# Reset style and plot Rating vs Gender
# plt.style.use("default")
# plt.figure(figsize=(5, 5))
# sns.barplot(x="Rating", y="Gender", data=numeric_df[170:180])
# plt.title("Rating vs Gender", fontsize=15)
# plt.xlabel("Rating")
# plt.ylabel("Gender")
# plt.show()

# Reset style and plot Rating vs Quantity
# plt.style.use("default")
# plt.figure(figsize=(5, 5))
# sns.barplot(x="Rating", y="Quantity", data=numeric_df[170:180])
# plt.title("Rating vs Quantity", fontsize=15)
# plt.xlabel("Rating")
# plt.ylabel("Quantity")
# plt.show()

""" # Product line Distribution
plt.figure(figsize=(12, 6))
productline = df["Product line"].value_counts().reset_index()
productline.columns = ['Product line', 'Count']  # Rename the columns

plt.pie(productline['Count'],
        labels=productline['Product line'], autopct='%1.1f%%')
plt.title("Product line Distribution")

plt.savefig('plots/product_line_distribution.png',
            format='png', dpi=300, bbox_inches='tight')

plt.show() """
