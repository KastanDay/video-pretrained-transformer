import requests
from PIL import Image
import torch
import json
import os
import pathlib
import sys
import numpy as np
import more_itertools
import pandas as pd

from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoImageProcessor, Mask2FormerModel
from transformers import AutoProcessor, VisionEncoderDecoderModel


class Vision_Experts:
    def __init__(self,device='cuda'):

        ### Setting the expert image processors and models
        self.device = device
        ## OCR Expert
        self.model_ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-str").to(device)
        self.processor_ocr = AutoProcessor.from_pretrained("microsoft/trocr-large-str")

        ## Object Expert
        self.model_obj = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)
        self.processor_obj = AutoProcessor.from_pretrained("google/owlvit-large-patch14")
        
        ## Segmentation Expert
        self.model_seg = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(device)
        self.processor_seg = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
       
        ## Depth Expert

        self.processor_depth =DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model_depth =DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)

    def get_object_embeddings(self,image):
        texts = ['Cat','Dog']                      ### Need to check if text this can be None
        inputs = self.processor_obj(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model_obj(**inputs,output_hidden_states=True)
        
        print(outputs['image_embeds'][0].shape)
        
        return outputs['image_embeds']
    
    def get_character_embeddings(self,image):
        pixel_values = self.processor_ocr(image, return_tensors="pt").to(self.device).pixel_values
        outputs = self.model_ocr(pixel_values=pixel_values,output_hidden_states=True)

        print(outputs['encoder_last_hidden_state'][0].shape)

        return outputs['encoder_last_hidden_state']
    
    def get_depth_embeddings(self,images):
        # prepare image for the model
        inputs = self.processor_depth(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_depth(**inputs,output_hidden_states=True)
        print(outputs['hidden_states'][0].shape)

        return outputs['hidden_states']
    
    def get_segmentation_embeddings(self,images):
   
        inputs = self.processor_seg(images, return_tensors="pt").to(self.device)
        # forward pass
        with torch.no_grad():
            outputs = self.model_seg(**inputs,output_hidden_states=True)
        print(outputs['encoder_last_hidden_state'][0].shape)

        return outputs['encoder_last_hidden_state']

    def pad_or_truncate_tensor(self, tensor, truncate_shape):
        target_shape = [truncate_shape, 1024]
        tensor_shape = tensor.shape

        # If tensor shape is larger than the target shape, truncate the tensor
        if tensor_shape[0] > target_shape[0]:
            truncated_tensor = tensor[:target_shape[0], :]
            return truncated_tensor

        # If tensor shape is smaller than the target shape, pad the tensor
        elif tensor_shape[0] < target_shape[0]:
            padding_shape = (target_shape[0] - tensor_shape[0], target_shape[1])
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_shape[0]), value=-100)
            return padded_tensor
        # If tensor shape is already the target shape, return the tensor
        else:
            return tensor

    def list_to_tensor(self, embed_list, max_encodings_to_make):
        # Converting a list of tensors to a tensor
        # Get the shape of the first tensor in the list
        tensor_shape = embed_list[0].shape
        # Create a tensor of zeros with the appropriate shape
        tensor = torch.zeros(len(embed_list), *tensor_shape)
        # Fill the tensor with the values from the input list
        for i, t in enumerate(embed_list):
            tensor[i, ...] = t
            fixed_tensor = self.pad_or_truncate_tensor(tensor, max_encodings_to_make)
            fixed_tensor.cpu()
        return fixed_tensor


    def get_vision_embed_from_vid_name(self,vid_name, max_encodings_to_make=220):

        # collect all frames from filepath
        segment_frames_paths = pathlib.Path(self.vid_name_to_frames_path(vid_name)).glob('*.jpg')
        frames_PIL_list = []
        for path in segment_frames_paths:
            if len(frames_PIL_list) >= max_encodings_to_make:
                break
            with Image.open(path) as img:
                frames_PIL_list.append(np.array(img))

        # split frames_PIL_list into batches of 100 (to avoid OOM)
        batch_list = list(more_itertools.batched(frames_PIL_list, 110))
        depth_embeddings = []
        objdet_embeddings = []
        segment_embeddings = []
        char_embeddings = []
        for batch in batch_list:
            depth_embeddings.extend(self.get_depth_embeddings(batch))
            segment_embeddings.extend(self.get_segmentation_embeddings(batch))
            objdet_embeddings.extend(self.get_object_embeddings(batch))
            char_embeddings.extend(self.get_character_embeddings(batch))

        depth_tensor = self.list_to_tensor(depth_embeddings,max_encodings_to_make)
        segment_tensor = self.list_to_tensor(segment_embeddings,max_encodings_to_make)
        objdet_tensor = self.list_to_tensor(objdet_embeddings,max_encodings_to_make)
        char_tensor = self.list_to_tensor(char_embeddings,max_encodings_to_make)

        return depth_tensor,segment_tensor,objdet_tensor,char_tensor


