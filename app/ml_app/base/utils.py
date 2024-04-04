from minio import Minio
import io
import pickle
import sklearn

def predict(values):
    minio_client = Minio(
        "192.168.1.23:9000",
        "01UEi1KBqoAFeN3ANxIK",
        "KXeCpTVnkovXYg1Mi3pSJrOkMlzNe5sbJiHdubxK",
        secure=False
        )

    bucket_name = 'premier-league'
    model_name = 'model.pkl'

    try:
        res = minio_client.get_object(bucket_name, model_name)

    except Exception as err:
        print(err)
    model = pickle.loads(res.data)

    result = model.predict([values])
    return result
