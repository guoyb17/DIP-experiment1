from requests import post
from requests.exceptions import ConnectionError
from json import dumps, loads
import os


def read_secret(token_file="API_SECRET"):
    f = open(token_file)
    api_key = f.readline().strip("\n")
    api_secret = f.readline().strip("\n")
    return (api_key, api_secret)

def get_mark(img_file):
    '''
    Call Face++ API.
    '''
    api_key, api_secret = read_secret()
    post_request = {
        "api_key": api_key,
        "api_secret": api_secret,
        "return_landmark": 2,
        # "image_file": open(img_file, "rb").read()
    }
    file_request = {
        "image_file": open(img_file, "rb")
    }
    fl = 0
    while fl == 0:
        try:
            response = post("https://api-cn.faceplusplus.com/facepp/v3/detect",
                            data=post_request, files=file_request)
            fl = 1
        except ConnectionError:
            continue
    content = response.content
    ret_data = loads(content)
    return ret_data["faces"][0]["landmark"]
