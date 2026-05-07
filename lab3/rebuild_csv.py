import json
import os
import csv

summaries_dir = "lab3_results/summaries"
json_files = sorted([f for f in os.listdir(summaries_dir) if f.endswith(".json")])

# Extract feature info from json files
rows = []
for json_file in json_files:
    if json_file == "lmfcc_hidden64_ep1.json":  # Skip smoke test
        continue
    with open(os.path.join(summaries_dir, json_file)) as f:
        data = json.load(f)
    rows.append({
        'feature_type': data['feature_type'],
        'hidden_layers': str(data['hidden_layers']),
        'hidden_size': data['hidden_size'],
        'epochs': data['epochs'],
        'train_accuracy': data['train_accuracy'],
        'validation_accuracy': data['validation_accuracy'],
        'test_accuracy': data['test_accuracy'],
        'train_loss': data['train_loss'],
        'validation_loss': data['validation_loss'],
        'test_loss': data['test_loss'],
        'state_accuracy': data['state_accuracy'],
        'phoneme_accuracy': data['phoneme_accuracy'],
        'state_per': data['state_per'],
        'phoneme_per': data['phoneme_per'],
        'model_path': data['model_path'],
        'posterior_plot_path': data['posterior_plot_path']
    })

# Write CSV in order: lmfcc, mspec, dlmfcc, dmspec
feature_order = {'lmfcc': 0, 'mspec': 1, 'dlmfcc': 2, 'dmspec': 3}
rows_sorted = sorted(rows, key=lambda x: feature_order.get(x['feature_type'], 999))

csv_path = os.path.join(summaries_dir, "lab3_section5_results.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows_sorted)

print(f"Rebuilt CSV with {len(rows_sorted)} experiments")
for row in rows_sorted:
    print(f"  {row['feature_type']}: test_acc={row['test_accuracy']:.4f}, state_PER={row['state_per']:.4f}")
