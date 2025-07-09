import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocess import load_and_clean_data
from src.model_utils import save_model

def main():
    df, label_encoders = load_and_clean_data("data/ITSM_data.csv")

    X = df.drop('Priority_Label', axis=1)
    y = df['Priority_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    save_model(clf, label_encoders)

    y_pred = clf.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("model/feature_importance.png")
    print("Training complete.")

if __name__ == "__main__":
    main()