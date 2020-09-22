from pycocotools.coco import COCO
#from matplotlib import pyplot as plt
#from matplotlib.patches import Rectangle
#import matplotlib.lines as mlines
from PIL import Image
import math
import itertools

class MsCocoDataset:
    def __init__(self):
        print("instance created")

        # Define location of annotations
        self.dataDir = '/data/share/frame_border_detection_db_v6/'
        self.db_type = "val"
        #dataType = 'val2017'
        dataType = str(self.db_type) + "2017"
        annFile = self.dataDir + "/instances_" + dataType + ".json"
        self.categories = ['person']  # , 'cat', 'dog'
        #self.categories = ['person', 'car', 'airplane', 'motorcycle', 'cat', 'dog', 'bench', 'elephant', 'laptop', 'backpack']

        # Create instance
        self.coco = COCO(annFile)

    def getImageIdsOfCategoryList(self, categories):
        # Filter for specific categories
        catIds = self.coco.getCatIds(catNms=categories)
        imgIds = self.coco.getImgIds(catIds=catIds)
        cat_names = self.coco.loadCats(catIds)
        return imgIds, cat_names, catIds

    def getAnnotationIdsOfImgId(self, image_id, categories_id):
        # firstImgId = imgIds_sub[:1]
        annotation_ids = self.coco.getAnnIds(image_id, categories_id)
        return annotation_ids

    def getBoundingBoxesOfAnnotationId(self, annotation_ids):
        bounding_box_list = self.coco.loadAnns(annotation_ids)
        return bounding_box_list;

    def downloadImagesOfCategory(self, categories):
        imgIds, cat_names, catIds = self.getImageIdsOfCategoryList(categories)
        print(len(imgIds))
        self.coco.download(self.dataDir + "/ms_coco_images_v3/" + str(self.db_type) + "/", imgIds)

        # firstImgId = img_id
        #filename = "/{:012d}".format(image_id)
        #dst_path = "/data/share/ms_coco/imgs/"
        #img_pil = Image.open(dst_path + filename + ".jpg", 'r')  # 12

    def run(self):
        self.downloadImagesOfCategory(self.categories)




