from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming you have ground truth labels and predicted labels
ground_truth_labels = [...]  # List of actual labels
predicted_labels = [...]     # List of predicted labels from the model

precision = precision_score(ground_truth_labels, predicted_labels, average='weighted')
recall = recall_score(ground_truth_labels, predicted_labels, average='weighted')
f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
