from minio import Minio

def get_res():
    minio_client = Minio(
        ":9000",
        "01UEi1KBqoAFeN3ANxIK",
        "KXeCpTVnkovXYg1Mi3pSJrOkMlzNe5sbJiHdubxK",
        secure=False
        )
    bucket_name = 'premier-league'
    thedataset = 'premier-league-tables.csv'

    try:
        res = minio_client.get_object(bucket_name, thedataset)
        print("GOT")
    except Exception as err:
        print("ERR")
        print(err)

get_res()
