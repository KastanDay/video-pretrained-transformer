import requests
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, OwlViTForObjectDetection
from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import AutoImageProcessor, Mask2FormerModel
from transformers import AutoProcessor, VisionEncoderDecoderModel


class Vision_Experts:
    def __init__(self):

        ### Setting the expert image processors and models

        ## OCR Expert
        self.model_ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-str")
        self.processor_ocr = AutoProcessor.from_pretrained("microsoft/trocr-large-str")

        ## Object Expert
        self.model_obj = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        self.processor_obj = AutoProcessor.from_pretrained("google/owlvit-large-patch14")
        
        ## Segmentation Expert
        self.model_seg = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        self.processor_seg = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
       
        ## Depth Expert

        self.processor_depth =DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model_depth =DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    def get_object_embeddings(self,image):
        texts = ['Cat','Dog']                      ### Need to check if text this can be None
        inputs = self.processor_obj(text=texts, images=image, return_tensors="pt")
        outputs = self.model_obj(**inputs,output_hidden_states=True)
        
        print(outputs['image_embeds'][0].shape)
        
        return outputs['image_embeds']
    
    def get_character_embeddings(self,image):
        pixel_values = self.processor_ocr(image, return_tensors="pt").pixel_values
        outputs = self.model_ocr(pixel_values=pixel_values,output_hidden_states=True)

        print(outputs['encoder_last_hidden_state'][0].shape)

        return outputs['encoder_last_hidden_state']
    
    def get_depth_embeddings(self,image):
        # prepare image for the model
        inputs = self.processor_depth(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model_depth(**inputs,output_hidden_states=True)
        print(outputs['hidden_states'][0].shape)

        return outputs['hidden_states']
    
    def get_segmentation_embeddings(self,image):
   
        inputs = self.processor_seg(image, return_tensors="pt")
        # forward pass
        with torch.no_grad():
            outputs = self.model_seg(**inputs,output_hidden_states=True)
        print(outputs['encoder_last_hidden_state'][0].shape)

        return outputs['encoder_last_hidden_state']
