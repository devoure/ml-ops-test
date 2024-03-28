from kfp.dsl import component, pipeline
import kfp

@component(
        packages_to_install=["pandas"],
        base_image="python:3.9"
        )
def load_data():
    import pandas as pd

    teams = pd.read_csv('/content/premier-league-tables.csv')
    data = pd.DataFrame(teams)
    data.to_pickle('./data.pkl')

@component(
        packages_to_install=["pandas", "scikit-learn"],
        base_image="python:3.9"
        )
def process_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_pickle('./data.pkl')
    del data["Notes"]

    def add_new_col(row):
        if row['Rk'] == 1:
            return 1
        else:
            return 0

    data['Won_League'] = data.apply(add_new_col, axis=1)
    X = data[['W', 'D', 'GD']]
    y = data['Won_League']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test.to_pickle('./X_test.pkl')
    X_train.to_pickle('./X_train.pkl')

    y_train.to_pickle('./y_train.pkl')
    y_test.to_pickle('./y_test/pkl')

@component(
        packages_to_install=["scikit-learn", "pandas"],
        base_image="python:3.9"
        )
def train_data():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import pickle

    X_train = pd.read_pickle('./X_train.pkl')
    y_train = pd.read_pickle('./y_train.pkl')

    model = LogisticRegression()
    model.fit(X_train, y_train)

    with open('./model.pkl', 'wb') as f:
        pickle.dump(model, f)

@pipeline(
        name="Premier League ML Pipeline",
        description="A simple pipeline to automate a machine learning workflow to create pl model"
        )
def pl_pipeline():
    load_data_component = load_data()
    process_data_component = process_data().after(load_data_component)
    train_data_component = train_data().after(process_data_component)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pl_pipeline, "pl_pipeline.yaml")
