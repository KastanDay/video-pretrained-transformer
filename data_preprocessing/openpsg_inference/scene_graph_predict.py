import os
import tempfile
import shutil
from typing import List

import sys
sys.path.append("/home/kastan/thesis/video-pretrained-transformer/data_preprocessing/openpsg_inference/original_psg/OpenPSG")
from cog import BasePredictor, Path, Input, BaseModel

from openpsg.utils.utils import show_result
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import mmcv


class ModelOutput(BaseModel):
    image: Path

MERGED_TO_NATURAL_LANG = {
    'tree-merged': 'group of trees',
    'fence-merged': 'fences',
    'ceiling-merged': 'ceiling',
    'sky-other-merged': 'the sky',
    'cabinet-merged': 'cabinets',
    'table-merged': 'a table',
    'floor-other-merged': 'the floor',
    'pavement-merged': 'the pavement',
    'mountain-merged': 'mountains',
    'grass-merged': 'grasses',
    'dirt-merged': 'dirt',
    'paper-merged': 'papers',
    'food-other-merged': 'bunch of food',
    'building-other-merged': 'group of buildings',
    'rock-merged': 'rocks',
    'wall-other-merged': 'walls',
    'rug-merged': 'a rug'
}

def get_scene_graph_list(tuple_list):
        output_string_list =[]
        for i in range (0,len(tuple_list)):
            subject_value = tuple_list[i][0]
            relation_value = tuple_list[i][1]
            object_value = tuple_list[i][2]
            if subject_value in MERGED_TO_NATURAL_LANG.keys():
                subject_value = MERGED_TO_NATURAL_LANG[subject_value]
            if object_value in MERGED_TO_NATURAL_LANG.keys():
                object_value = MERGED_TO_NATURAL_LANG[object_value]
            final_string = subject_value + ' ' +relation_value + ' ' + object_value
            output_string_list.append(final_string)

        return output_string_list

class Predictor(BasePredictor):
    def __init__(self):
        # model_ckt = "epoch_60.pth"
        model_ckt = "checkpoints/epoch_60.pth"
        cfg = Config.fromfile("configs/psgtr/psgtr_r50_psg_inference.py")
        self.model = init_detector(cfg, model_ckt, device="cuda")

    def predict(
        self,
        # image: Path = Input(
        #     description="Input image.",
        # ),
        image = Input(
            description="Input image, as CV2 file in-memory.",
        ),
        num_rel: int = Input(
            description="Number of Relations. Each relation will generate a scene graph",
            default=5,
            ge=1,
            le=20,
        ),
    ) -> List[ModelOutput]:
        input_image = mmcv.imread(image)
        
        result = inference_detector(self.model, input_image)
        # out_path = "data/simple_test_data/ouput.png"
        # out_dir = "data/simple_test_data/"

        sro_tuple_list = show_result(
            # str(image), # 
            image,
            result,
            is_one_stage=True,
            num_rel=num_rel,
            # out_dir=out_dir,
            # out_file=str(out_path),
        )
        scg_list = get_scene_graph_list(sro_tuple_list)

        return scg_list  ## returning the list of strings for the scene graphs


def print_debugging():
    pass
    # output = []
    # print(ModelOutput(image=out_path))
    # output.append(ModelOutput(image=out_path)) ## We get a not valid URL error for ModelOutput!!!
    # print("Out path", out_dir)
    # for i, img_path in enumerate(os.listdir(out_dir)):
    #     img = mmcv.imread(os.path.join(out_dir, img_path))
    #     out_path = f"output_{i}.png"
    #     mmcv.imwrite(img, str(out_path))
    #     # output.append(ModelOutput(image=out_path))
    #     output.append(out_path)
    # shutil.rmtree(out_dir)
