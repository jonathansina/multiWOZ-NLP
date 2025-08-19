import numpy as np
import matplotlib.pyplot as plt


def find_zero_percentage(loader, name, max_length):
    return np.mean(
        [
            (loader.dataset[i][name] == 0).sum() / max_length
            for i in range(len(loader.dataset))
        ]
    )


def plot_model_performance(train_evaluation, valid_evaluation):
    plt.figure(figsize=(10, 6))

    metric_names = list(train_evaluation.keys())
    
    scores = {}
    for metric in metric_names:
        scores[metric] = [train_evaluation[metric], valid_evaluation[metric]]

    labels = ['Train', 'Validation']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    x = np.arange(len(labels))
    width = 0.8 / len(metric_names)
    bars = []
    for i, metric in enumerate(metric_names):
        bar = plt.bar(
            x + (i - len(metric_names)/2 + 0.5) * width,
            scores[metric], 
            width, 
            label=metric.replace('_', ' ').title(), 
            color=colors[i % len(colors)], 
            alpha=0.8
        )
        bars.extend(bar)

    plt.xlabel('Dataset Split')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(ticks=x, labels=labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f'{height:.4f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), 
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10
        )

    plt.tight_layout()