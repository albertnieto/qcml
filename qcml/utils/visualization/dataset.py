import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb
from qic.utils.visualization.seaborn import palette_saturated, adjust_saturation

def plot_decision_boundary(X, y, model, ax, palette, saturation_factor=1.5):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Adjust saturation of the palette
    saturated_palette = adjust_saturation(palette, factor=saturation_factor)
    
    cmap = ListedColormap(saturated_palette)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(y.max() + 2) - 0.5, cmap=cmap)

def plot_2d_dataset(train_csv, test_csv, feature1_index=0, feature2_index=1, axis_titles=None, palette=None, fig_size=(4, 3), decision_boundary=None, saturation_factor=1.5):
    df_train = pd.read_csv(train_csv, header=None)
    df_test = pd.read_csv(train_csv, header=None)
    
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    
    class_col = df_combined.columns[-1]
    features = df_combined.iloc[:, :-1]
    
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        df_combined = pd.DataFrame(features_pca, columns=['PCA1', 'PCA2'])
        df_combined['Class'] = pd.concat([df_train.iloc[:, -1], df_test.iloc[:, -1]], ignore_index=True)
    else:
        df_combined = pd.concat([features, df_combined[class_col]], axis=1)
        df_combined.columns = ['Feature1', 'Feature2', 'Class']
    
    X = df_combined.iloc[:, :2].values
    y = df_combined['Class'].values
    
    if palette is None:
        unique_classes = np.unique(y).size
        palette = palette_saturated(palette_name='coolwarm', n_colors=unique_classes)
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot decision boundary if a classifier is provided
    if decision_boundary is not None:
        decision_boundary.fit(X, y)
        plot_decision_boundary(X, y, decision_boundary, ax, palette, saturation_factor=saturation_factor)
    
    # Plot the data points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=palette,
        ax=ax,
        edgecolor='k'
    )
    
    if axis_titles:
        ax.set_xlabel(axis_titles[0])
        ax.set_ylabel(axis_titles[1])
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    ax.legend().remove()  # No legend by default
    
    plt.show()
