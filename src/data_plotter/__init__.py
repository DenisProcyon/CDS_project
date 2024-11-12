import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataPlotter:
    def __init__(self) -> None:
        sns.set_theme(style="whitegrid") 

    @property
    def default_configs(self):
        return {
            "figsize": (12, 8),
            "color": "skyblue",       
            "title_fontsize": 16,    
            "label_fontsize": 14,      
            "tick_labelsize": 12       
        }

    def plot_barchart(self, data: pd.Series, title: str, xlabel: str, ylabel: str, figsize: tuple = None):
        plt.figure(figsize=self.default_configs["figsize"] if figsize is None else figsize)

        ax = data.plot(kind='bar', color=self.default_configs["color"], edgecolor='black')

        plt.title(title, fontsize=self.default_configs["title_fontsize"], weight='bold')
        plt.xlabel(xlabel, fontsize=self.default_configs["label_fontsize"])
        plt.ylabel(ylabel, fontsize=self.default_configs["label_fontsize"])

        plt.xticks(rotation=45, fontsize=self.default_configs["tick_labelsize"], ha='right')
        plt.yticks(fontsize=self.default_configs["tick_labelsize"])

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout() 
        plt.show()
