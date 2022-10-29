import os
import tempfile
import shutil
from typing import List
from cog import BasePredictor, Path, Input, BaseModel

from openpsg.utils.utils import show_result
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import mmcv


class ModelOutput(BaseModel):
    image: Path


# todo!
MERGED_TO_NATURAL_LANG = {
    'tree-merged': 'group of trees', 
    'fence-merged': 'fences', 
    'ceiling-merged': 'ceiling', 
    'sky-other-merged': 'the sky',
    'cabinet-merged': 'cabinets',
    'table-merged': 'a table', 
    'floor-other-merged': '', 
    'pavement-merged': '',
    'mountain-merged': '', 
    'grass-merged': '', 
    'dirt-merged': '', 
    'paper-merged': '',
    'food-other-merged': '', 
    'building-other-merged': '', 
    'rock-merged': '',
    'wall-other-merged': '', 
    'rug-merged': ''
}

class Predictor(BasePredictor):
    def setup(self):
        # model_ckt = "epoch_60.pth"
        model_ckt = "checkpoints/epoch_60.pth"
        cfg = Config.fromfile("configs/psgtr/psgtr_r50_psg_inference.py")
        self.model = init_detector(cfg, model_ckt, device="cuda")

    # todo: Rohan.
    def get_scene_graph_list(self, path_to_img: str ='/home/kastan/thesis/data/simple_test_data/test_img_2.png', num_relations: int = 10):
        '''
        path_to_img: path to image
        num_rel: int 
        Return type: 
        [
            
            'person beside flower',
            'person under the sky',
        ]
        '''
        raise NotImplementedError # todo remove
        
        # something like this...
        for object_str in all_results:
            if object_str in MERGED_TO_NATURAL_LANG.keys():
                object_str = MERGED_TO_NATURAL_LANG[object_str]
        
        # todo 
        # return a list of triplets, but each triplet is a string.
        return results
    
    def predict(
        self,
        image: Path = Input(
            description="Input image.",
        ),
        num_rel: int = Input(
            description="Number of Relations. Each relation will generate a scene graph",
            default=5,
            ge=1,
            le=20,
        ),
    ) -> List[ModelOutput]:
        input_image = mmcv.imread(str(image))
        print("Image path", str(image))
        result = inference_detector(self.model, input_image)
        out_path = "data/simple_test_data/ouput.png"
        out_dir = "data/simple_test_data/"
        print("result.rels", result.rels)
        show_result(
            str(image),
            result,
            is_one_stage=True,
            num_rel=num_rel,
            out_dir=out_dir,
            out_file=str(out_path),
        )
        output = []
        # print(ModelOutput(image=out_path))
        # output.append(ModelOutput(image=out_path)) ## We get a not valid URL error for ModelOutput!!!
        print("Out path", out_dir)
        for i, img_path in enumerate(os.listdir(out_dir)):
            img = mmcv.imread(os.path.join(out_dir, img_path))
            out_path = f"output_{i}.png"
            mmcv.imwrite(img, str(out_path))
            # output.append(ModelOutput(image=out_path))
            output.append(out_path)
        shutil.rmtree(out_dir)

        return output

my_pred = Predictor()
my_pred.setup()
output = my_pred.predict(image="/home/kastan/thesis/data/simple_test_data/test_img_2.png",num_rel = 15)
print("Final output", output)
