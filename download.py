import argparse
import os
import re
from tqdm import tqdm
import requests
import json, argparse

sess = requests.Session()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link', '-l', type=str, required=True, help='Share link of Tsinghua Cloud')
    parser.add_argument('--password', '-p', type=str, default='', help='Password of the share link')
    parser.add_argument('--save', '-s', type=str, default='./', help='Save directory')
    parser.add_argument('--file', '-f', type=str, default=None, help='File name, support regex, if not set, download all files')
    return parser.parse_args()

def get_share_key(url):
    prefix = 'https://cloud.tsinghua.edu.cn/d/'
    if not url.startswith(prefix):
        raise ValueError('Share link of Tsinghua Cloud should start with {}'.format(prefix))
    share_key = url[len(prefix):].replace('/', '')     
    print('Share key: {}'.format(share_key))
    
    return share_key
        
    
def dfs_search_files(share_key: str, path="/"):
    global sess
    filelist = []
    print('https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{}/dirents/?path={}'.format(share_key, path))
    r = sess.get('https://cloud.tsinghua.edu.cn/api/v2.1/share-links/{}/dirents/?path={}'.format(share_key, path))
    objects = r.json()['dirent_list']
    for obj in objects:
        if obj["is_dir"]:
            filelist += dfs_search_files(share_key, obj['folder_path'])
        else:
            filelist.append(obj)

    return filelist
    
def download_single_file(url: str, fname: str):
    global sess
    resp = sess.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    dir_name = os.path.dirname(fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(fname, 'wb') as file, tqdm(
        total=total,
        ncols=120,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download(url, save_dir):
    share_key = get_share_key(url)
    
    print("Searching for files to be downloaded...")
    search_files = dfs_search_files(share_key)
    # for file in search_files:
    #     print(file['is_dir'], file.keys())
    filelist = sorted(search_files, key=lambda x: x['file_path'])
    print("Found {} files in the share link.".format(len(filelist)))
    print("Last Modified Time".ljust(25), " ", "File Size".rjust(10), " ", "File Path")
    print("-" * 100)
    for file in filelist:
        print(file["last_modified"], " ", str(file["size"]).rjust(10), " ", file["file_path"])
    print("-" * 100)
    
    if not args.yes:
        while True:
            key = input("Start downloading? [y/n]")
            if key == 'y':
                break
            elif key == 'n':
                return
    
    flag = True
    for i, file in enumerate(filelist):
        file_url = 'https://cloud.tsinghua.edu.cn/d/{}/files/?p={}&dl=1'.format(share_key, file["file_path"])
        save_path = os.path.join(save_dir, file["file_path"][1:])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("[{}/{}] Downloading File: {}".format(i + 1, len(filelist), save_path))
        try:
            download_single_file(file_url, save_path)
        except Exception as e:
            print("Error happened when downloading file: {}".format(save_path))
            print(e)
            flag = False
    if flag:
        print("Download finished.")
    else:
        print("Download finished with error.")
    
    return flag

def make_data(sample):
    src = ""
    for ix, ref in enumerate(sample['references']):
        src += "Reference [%d]: %s\\" % (ix+1, ref)
    src += "Question: %s\\Answer:" % (sample['question'])
    source = src.replace("\n", " ").replace("\r", " ")
    target = sample['answer'].replace("\n"," ").replace("\r", " ")
    
    return source, target
    
if __name__ == "__main__":
    
    arg = argparse.ArgumentParser()
    arg.add_argument('target', type=str, choices=["generator-training-data", "retriever-training-data", "retriever-pretrained-checkpoint", "all"], help='Target to download')
    arg.add_argument('--save', '-s', type=str, default='./download', help='Save directory')
    arg.add_argument("-y", "--yes", action="store_true", help="Download without confirmation")
    args = arg.parse_args()
    
    if args.target in ["all", "generator-training-data"]:
        
        save_dir = os.path.join(args.save, 'generator-training-data', 'raw')
        if download('https://cloud.tsinghua.edu.cn/d/d290dcfc92e342f9a017/', save_dir):
            
            for split in ['train', 'val', 'test']:
                ds = [json.loads(data) for data in open(f'{save_dir}/{split}.jsonl').readlines()]
                processed_dir = os.path.join(args.save, 'generator-training-data', 'processed')
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir)
                source_out = open(os.path.join(processed_dir, f'{split}.source'), 'w')
                target_out = open(os.path.join(processed_dir, f'{split}.target'), 'w')
                for sample in tqdm(ds):
                    source, target = make_data(sample)
                    source_out.write(source + '\n')
                    target_out.write(target + '\n')
            
                source_out.close()
                target_out.close()
            
    if args.target in ["all", "retriever-training-data"]:
        download("https://cloud.tsinghua.edu.cn/d/3927b67a834c475288e2/", os.path.join(args.save, 'retriever-training-data'))
        
    if args.target in ["all", "retriever-pretrained-checkpoint"]:
        download("https://cloud.tsinghua.edu.cn/d/bc96946dd9a14c84b8d4/", os.path.join(args.save, 'retriever-pretrained-checkpoint'))  
