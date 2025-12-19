import matplotlib.pyplot as plt
import seaborn as sns
class Ploter:
    def __init__(self):
        pass
    def count_ploter(self, column,data):
        sns.countplot(x=column, data=data)
        plt.title(f"Class Distribution of {column}")
        plt.show()
    def hist_ploter(self, data_col):
        sns.histplot(data_col, bins=30, kde=True)
        plt.title(f"{data_col.name} Distribution")
        plt.show()
    def box_ploter(self ,column_x, column_y, data):
        sns.boxplot(x=column_x, y=column_y, data=data)
        plt.title(f"{column_y} by {column_x}")
        plt.show()
    def bar_ploter(self, data_col):
        sns.barplot(x=data_col.index, y=data_col.values)
        plt.title(f"Bar Plot of {data_col.name}")
        plt.show()