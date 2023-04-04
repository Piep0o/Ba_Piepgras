from img2vec_pytorch import Img2Vec
from PIL import Image
from tqdm.auto import tqdm
import os
import pandas as pd
# Initialize Img2Vec
img2vec = Img2Vec()

### takes img_path as input and returns img vector
def return_embedding(img_path):
    img = Image.open(img_path)
    vec = img2vec.get_vec(img)
    curr_df = pd.DataFrame(vec).T
    return curr_df

k = 0

imgs_path = os.listdir('food-101_/images/')
embedding_df = pd.DataFrame()

### looping over img in img_path
for curr_img in tqdm(imgs_path):

    try:
        curr_df = return_embedding('food-101_/images/' + str(curr_img))
        curr_df['img'] = curr_img
        embedding_df = pd.concat([embedding_df, curr_df])
    except:
        print("Something went wrong with an img")
        continue


### creating csv file to work with
embedding_df.to_csv("foo.csv", mode='w', index=False)
