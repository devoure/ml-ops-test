from kfp.dsl import component, pipeline
import kfp

@component(
        base_image="devourey/scikit-sandbox:latest"
        )
def load_data():
    import pandas as pd
    from minio import Minio
    import io

    minio_client = Minio(
        "192.168.1.29:9000",
        "01UEi1KBqoAFeN3ANxIK",
        "KXeCpTVnkovXYg1Mi3pSJrOkMlzNe5sbJiHdubxK",
        secure=False
        )

    bucket_name = 'premier-league'
    thedataset = 'premier-league-tables.csv'

    try:
        res = minio_client.get_object(bucket_name, thedataset)

    except Exception as err:
        print(err)

    teams = pd.read_csv(io.BytesIO(res.data))
    data = pd.DataFrame(teams)

    encodeddata = data.to_csv(index=False).encode("utf-8")

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        minio_client.put_object(
                bucket_name,
                'data.csv',
                data=io.BytesIO(encodeddata),
                length=len(encodeddata),
                content_type='application/csv'
                )
    except Exception as err:
        print(err)

@component(
        base_image="devourey/scikit-sandbox:latest"
        )
def process_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from minio import Minio
    import io

    minio_client = Minio(
        "192.168.1.29:9000",
        "01UEi1KBqoAFeN3ANxIK",
        "KXeCpTVnkovXYg1Mi3pSJrOkMlzNe5sbJiHdubxK",
        secure=False
        )

    bucket_name = 'premier-league'
    try:
        res = minio_client.get_object(bucket_name, 'data.csv')

    except Exception as err:
        print(err)
    
    data = pd.read_csv(io.BytesIO(res.data))
    data = pd.DataFrame(data)

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

    X_train_bytes = X_train.to_csv(index=False).encode("utf-8")
    X_test_bytes = X_test.to_csv(index=False).encode("utf-8")
    y_train_bytes = y_train.to_csv(index=False).encode("utf-8")
    y_test_bytes = X_test.to_csv(index=False).encode("utf-8")

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
        minio_client.put_object(
                bucket_name,
                'X-train.csv',
                data=io.BytesIO(X_train_bytes),
                length=len(X_train_bytes),
                content_type = "application/csv"
                )
        minio_client.put_object(
                bucket_name,
                'X-test.csv',
                data=io.BytesIO(X_test_bytes),
                length=len(X_test_bytes),
                content_type = "application/csv"
                )
        minio_client.put_object(
                bucket_name,
                'y-train.csv',
                data=io.BytesIO(y_train_bytes),
                length=len(y_train_bytes),
                content_type = "application/csv"
                )
        minio_client.put_object(
                bucket_name,
                'y-test.csv',
                data=io.BytesIO(y_test_bytes),
                length=len(y_test_bytes),
                content_type = "application/csv"
                )

    except Exception as err:
        print(err)


@component(
        base_image="devourey/scikit-sandbox:latest"
        )
def train_data():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import pickle
    from minio import Minio
    import io

    minio_client = Minio(
        "192.168.1.29:9000",
        "01UEi1KBqoAFeN3ANxIK",
        "KXeCpTVnkovXYg1Mi3pSJrOkMlzNe5sbJiHdubxK",
        secure=False
        )

    bucket_name = 'premier-league'
    try:
        res_for_X = minio_client.get_object(bucket_name, 'X-train.csv')
        res_for_y = minio_client.get_object(bucket_name, 'y-train.csv')

    except Exception as err:
        print(err)

    X_train = pd.read_csv(io.BytesIO(res_for_X.data))
    y_train = pd.read_csv(io.BytesIO(res_for_y.data))
    
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    try:
        bytes_file = pickle.dumps(model)
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name='model.pkl',
            data=io.BytesIO(bytes_file),
            length=len(bytes_file)
        )
    except Exception as e:
        print(e)


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
