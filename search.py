import requests

from clip_retrieval.clip_client import ClipClient, Modality
from PIL import Image

from utils import get_image_name, get_new_image_name, prompts


def download_image(img_url, img_path):
    img_stream = requests.get(img_url, stream=True)
    if img_stream.status_code == 200:
        img = Image.open(img_stream.raw)
        img.save(img_path, format="png")
        return img_path


def download_best_available(search_result, result_img_path):
    if search_result:
        img_path = download_image(search_result[0]["url"], result_img_path)
        return img_path if img_path else download_best_available(search_result[1:], result_img_path)


class SearchSupport:
    def __init__(self):
        self.client = ClipClient(
            url="https://knn.laion.ai/knn-service",
            indice_name="laion5B-L-14",
            modality=Modality.IMAGE,
            aesthetic_score=0,
            aesthetic_weight=0.0,
            num_images=10,
        )


class ImageSearch(SearchSupport):
    def __init__(self, *args, **kwargs):
        print("Initializing Image Search")
        super().__init__()

    @prompts(name="Search Image That Matches User Input Text",
             description="useful when you want to search an image that matches a given description. "
                         "like: find an image that contains certain objects with certain properties, "
                         "or refine a previous search with additional criteria. " 
                         "The input to this tool should be a string, representing the description. ")
    def inference(self, query_text):
        search_result = self.client.query(text=query_text)
        return download_best_available(search_result, get_image_name())


class VisualSearch(SearchSupport):
    def __init__(self, *args, **kwargs):
        print("Initializing Visual Search")
        super().__init__()

    @prompts(name="Search Image Visually Similar to an Input Image",
             description="useful when you want to search an image that is visually similar to an input image. "
                         "like: find an image visually similar to a generated or modified image. "
                         "The input to this tool should be a string, representing the input image path. ")
    def inference(self, query_img_path):
        search_result = self.client.query(image=query_img_path)
        return download_best_available(search_result, get_new_image_name(query_img_path, "visual-search"))
