from wordcloud import WordCloud
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from ast import literal_eval
from pathlib import Path
import pathlib as pl 
import seaborn as sns
import pandas as pd
import os

from src.paths import LOCAL_DATA_PATH

possible_labels = {
    "per:positive_impression",
    "per:negative_impression",
    "per:acquaintance",
    "per:alumni",
    "per:boss",
    "per:subordinate",
    "per:client",
    "per:dates",
    "per:friends",
    "per:girl/boyfriend",
    "per:girl_boyfriend",
    "per:neighbor",
    "per:roommate",
    "per:children",
    "per:other_family",
    "per:parents",
    "per:siblings",
    "per:spouse",
    "per:place_of_residence",
    "per:place_of_birth",
    "per:visited_place",
    "per:origin",
    "per:employee_or_member_of",
    "per:schools_attended",
    "per:works",
    "per:age",
    "per:date_of_birth",
    "per:major",
    "per:place_of_work",
    "per:title",
    "per:alternate_names",
    "per:pet",
    "gpe:residents_of_place",
    "gpe:births_in_place",
    "gpe:visitors_of_place",
    "org:employees_or_members",
    "org:students",
    "unanswerable",
    "no_relation",
    "null_relation"
    }

possible_labels.update({label.split(':')[1] if ':' in label else label for label in possible_labels})


def get_report_paths(path):
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'report.json':
                paths.append(pl.Path(os.path.join(root, file)))
                
    return paths

# Initial explode
def get_labels(df, col):
    df_exploded = df[col].explode()

    # Loop to further explode nested lists
    while df_exploded.apply(lambda x: isinstance(x, list) or (isinstance(x, str) and x.startswith('['))).any():
        # Apply literal_eval only if the item is a string representation of a list
        df_exploded = df_exploded.apply(lambda x: literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        df_exploded = df_exploded.explode()

    return df_exploded

def plot_word_cloud(label_counts, possible_true_labels, output_path):
    # Creating a color function
    def color_func(word, *args, **kwargs):
        return 'green' if word in possible_true_labels else 'red'

    # Generating the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=color_func)
    wordcloud.generate_from_frequencies(label_counts)

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Predicted Labels')

    # Save the plot to the specified path
    wordcloud_output_path = output_path.parent / 'label_distribution_wordcloud.png'
    plt.savefig(wordcloud_output_path, bbox_inches='tight', dpi=600)  

    print(f"Word Cloud chart exported to\n{wordcloud_output_path}")
    plt.close()



def plot_label_distribution(p):
    # Read the JSON data
    df = pd.read_json(p, lines=True)

    # Processing the data
    if 'clsTskOnl' in str(p):
        predicted_labels = get_labels(df, 'predicted_labels')
        # possible_true_labels = get_labels(df, 'true_labels').unique().tolist()
    else:
        predicted_labels = df.predicted_labels.explode().dropna().apply(lambda x: x.get('relation', x.get('r')))
        # possible_true_labels = df.true_labels.explode().dropna().apply(lambda x: x.get('relation', x.get('r'))).unique().tolist()

    possible_true_labels = possible_labels

    # Counting the occurrences of each predicted label
    label_counts = predicted_labels.value_counts()

    # Calculate the dynamic height based on the number of labels
    min_height_per_label = 0.5  # Minimum height allocated for each label
    total_labels = len(label_counts)
    dynamic_height = total_labels * min_height_per_label

    # Creating the horizontal bar chart with dynamic figure size
    fig, ax = plt.subplots(figsize=(10, dynamic_height))
    sns.barplot(x=label_counts, y=label_counts.index, palette=['green' if label in possible_true_labels else 'red' for label in label_counts.index], ax=ax)
    plt.xlabel('Count')
    plt.ylabel('Predicted Labels')

    # Annotating the count on each bar in gray
    for index, (label, value) in enumerate(label_counts.items()):
        plt.text(value, index, str(value), color='gray')
        ax.get_yticklabels()[index].set_color('red' if label not in possible_true_labels else 'black')

    # Add legend
    green_patch = mpatches.Patch(color='green', label='Possible Labels')
    red_patch = mpatches.Patch(color='red', label='Hallucinated Labels')
    plt.legend(handles=[green_patch, red_patch], loc='lower right')
    plt.title('Predicted Label Distribution')

    # Save the plot to the specified path
    output_path = Path(p).parent / 'label_distribution.png'
    plt.savefig(output_path, bbox_inches='tight')

    print(f"Chart exported to\n{output_path}")
    # Close the plot to free up memory
    plt.close()
    
    return label_counts, possible_true_labels


if __name__ == "__main__":
    reports_path = LOCAL_DATA_PATH / 'reports'
    report_paths = get_report_paths(reports_path)

    for p in report_paths:
        df = pd.read_json(p, lines=True)
        label_counts, possible_true_labels = plot_label_distribution(p)
        plot_word_cloud(label_counts, possible_true_labels, Path(p))

