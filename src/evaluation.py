from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from model_training import train_models


def evaluate():
    X_test, y_test, lr, rf = train_models()

    models = {
        "Logistic Regression": lr,
        "Random Forest": rf
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)

        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(name)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    feature_importance = rf.feature_importances_
    features = X_test.columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    print("\nTop churn drivers:")
    print(importance_df.head(10))


if __name__ == "__main__":
    evaluate()
