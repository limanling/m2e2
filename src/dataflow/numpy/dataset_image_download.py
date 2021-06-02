import os
import argparse
import urllib.request
import json

def download_image(url_image, path_save):
    urllib.request.urlretrieve(url_image, path_save)

def download_image_list(meta_json, dir_save):
    if not os.path.exists(meta_json):
        print('[ERROR] input_metadata_json does not exist.')
    metadata = json.load(open(meta_json))
    for doc_id in metadata:
        for img_id in metadata[doc_id]:
            url_image = metadata[doc_id][img_id]['url']
            image_path_save = os.path.join(dir_save, '%s_%s.jpg' % (doc_id, img_id))
            if not url_image.endswith('.jpg'):
                print(image_path_save, url_image)
            download_image(url_image, image_path_save)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_metadata_json', type=str,
                        help='input_metadata_json')
    parser.add_argument('output_image_dir', type=str,
                        help='output_image_dir')
    args = parser.parse_args()
    
    input_metadata_json = args.input_metadata_json
    output_image_dir = args.output_image_dir

    download_image_list(input_metadata_json, output_image_dir)