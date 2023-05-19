import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def show_evaluation_page():

    # Define the evaluation metrics
    precision = [0.86, 0.86]
    recall = [0.91, 0.80]
    f1_score = [0.88, 0.83]
    support = [600, 431]
    accuracy = 0.86

    # Calculate macro average
    macro_precision = sum(precision) / len(precision)
    macro_recall = sum(recall) / len(recall)
    macro_f1_score = sum(f1_score) / len(f1_score)

    # Calculate weighted average
    weighted_precision = (precision[0] * support[0] + precision[1] * support[1]) / sum(support)
    weighted_recall = (recall[0] * support[0] + recall[1] * support[1]) / sum(support)
    weighted_f1_score = (f1_score[0] * support[0] + f1_score[1] * support[1]) / sum(support)

    # Define the confusion matrix
    confusion_matrix = [[544, 56],
                        [87, 344]]

    # Define the target names
    target_names = ['POSITIVE', 'NEGATIVE']
 # Create a function to show the confusion matrix
    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')
        st.pyplot(plt.gcf())

    # Display the evaluation metrics
    st.subheader('Evaluation Metrics')
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    }, index=target_names)
    metrics_df.loc['macro avg'] = [macro_precision, macro_recall, macro_f1_score, '']
    metrics_df.loc['weighted avg'] = [weighted_precision, weighted_recall, weighted_f1_score, '']
    st.write(metrics_df)

    # Display the accuracy
    st.subheader('Accuracy')
    st.write(f'Accuracy: {accuracy}')

    # Display the confusion matrix
    st.subheader('Confusion Matrix')
    cm_df = pd.DataFrame(confusion_matrix, index=target_names, columns=target_names)
    show_confusion_matrix(cm_df)


if __name__ == '__main__':
    show_evaluation_page()