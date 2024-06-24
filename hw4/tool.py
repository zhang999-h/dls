import os
import requests

# 基本设置
ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
data_dir = './data/ptb'

# 确保目标目录存在
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 下载文件
for f in ['train.txt', 'test.txt', 'valid.txt']:
    file_path = os.path.join(data_dir, f)
    if not os.path.exists(file_path):
        print(f"Downloading {f}...")
        response = requests.get(ptb_data + f)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"{f} downloaded.")
        else:
            print(f"Failed to download {f}. HTTP Status Code: {response.status_code}")
    else:
        print(f"{f} already exists.")