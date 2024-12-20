from data_loader import load_image_data
from preprocessing import preprocess_images
from model import train_model, evaluate_model
from utils import plot_results

def main():
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_image_data("dataset")

    print("Preprocessing images...")
    X_train, X_test = preprocess_images(X_train, X_test)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    print("Plotting results...")
    plot_results(model, X_test, y_test)

if __name__ == "__main__":
    main()
