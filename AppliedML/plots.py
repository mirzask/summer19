
# Plot only float64 data
sns.pairplot(car_data.loc[:,car_data.dtypes == 'float64'])
