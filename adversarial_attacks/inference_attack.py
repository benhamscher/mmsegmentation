from argparse import ArgumentParser

import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torch.nn.functional as F
from mmseg.apis import init_model
from mmengine.dataset import Compose
import mmcv

from attacks import Attacks
from torchvision.transforms.functional import to_pil_image

from LoadImageFromFile_512x1024 import LoadImageFromFile_512x1024

from mmseg.apis.inference import _preprare_data

import torch



import mmcv
#from .base import BaseTransform
#from .builder import TRANSFORMS

colors = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (0, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]

classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),

def assert_image_shape(image):
    assert image.ndimension() == 4, "Das Bild sollte 4 Dimensionen haben (batch, Channels, Height, Width)"
    assert image.size(0) == 1, "es muss 1 bild i, Batch sein"
    assert image.size(1) == 3, "Das Bild sollte 3 Kanäle haben (RGB)"
    print("Die Bildform ist korrekt!")

trans = transforms.Compose([transforms.ToTensor(),
                           
                            ])

def read_target(target_name, data_path, attack):

    city = target_name.split("_")[0]
    data_path = data_path.rstrip('/')

    if target_name[-3::] == "png":
        path = os.path.join(data_path, 'gtFine', "train", city, target_name.replace('leftImg8bit', 'gtFine_labelTrainIds'))
    else:
        path = os.path.join(data_path, 'gtFine', "train", city, target_name.replace('leftImg8bit', 'gtFine_labelTrainIds') + ".png")

    target = Image.open(path)
    target = np.array(target)
    target = trans(target) * 255
    target = target.unsqueeze(1)
    target = F.interpolate(target, size=(512, 1024), mode='nearest')
    target = target.squeeze(1)

    #if attack.lower() in ("dag", "pgd", 'alma_prox'):
        #target[target == 255] = 0
    #else: 
    target[target == 255] = -1
    
    target = target.long()
    return target


""""
def test():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"Current device number is: {torch.cuda.current_device()}, "
            f"device name {torch.cuda.get_device_name(torch.cuda.current_device())}")
"""

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) 
        #img = F.interpolate(img, size=(512, 1024), mode='bilinear', align_corners=True)
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class Cityscapes():

    def __init__(self, data_path, data_pipeline, split='val'):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """

        test_pipeline = [LoadImage()] + data_pipeline
        
        self.test_pipeline = Compose(test_pipeline)

        self.images = []  # where to load input images - absolute paths
        self.targets = []  # where to load ground truth if available - absolute paths
        self.filename = []  # image filename
        self.path = []  ##########

        for city in sorted(os.listdir(os.path.join(data_path, 'leftImg8bit', split))):
            for img in sorted(os.listdir(os.path.join(data_path, 'leftImg8bit', split, city))):
                self.images.append(os.path.join(data_path, 'leftImg8bit', split, city, img))
                self.targets.append(
                    os.path.join(data_path, 'gtFine', split, city, img.replace('leftImg8bit', 'gtFine_labelTrainIds')))
                self.filename.append(img.split('_leftImg8bit')[0])
                if split == 'val':
                    self.path.append(os.path.join(data_path, 'leftImg8bit', split, city, img))
                if split == 'train':
                    self.path.append(os.path.join(data_path, 'leftImg8bit', split, city, img))

    def __getitem__(self, index):
        """Generate one sample of data"""

    
        target = Image.open(self.targets[index])
        target = np.array(target)
        target = trans(target) * 255
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(512, 1024), mode='nearest')
        target = target.squeeze(1)
        target[target == 255] = -1
        target = target.long()

        #print("target shape", target.shape)
        image = None
        if self.path != []:
            return image, target, self.filename[index], self.path[index]
        else:
            return image, target, self.filename[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)


def main():
    print("___ run ___")

    parser = ArgumentParser()
    parser.add_argument('--img-path', help='Image path')  # ..... cityscapes
    parser.add_argument('--config', help='Config file')  # deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--out-path', default=None, help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--dataset', default='cityscapes', help='Dataset')  # cityscapes
    parser.add_argument('--attack', default='FGSM_untargeted', help='Type of attack')
    parser.add_argument('--modelname', default='FGSM_untargeted', help='Type of attack')
    parser.add_argument('--epsilon', default='2', help="max size of noise")
    parser.add_argument('--target', default=None, help="target")

    args = parser.parse_args()



    #### choose and define ####
    flag_save_raw_probs = False
    eps_value = int(args.epsilon)
    if args.target:
        target = read_target(args.target, args.img_path, args.attack)
    device = "cuda:0"
    


    

    if args.dataset == 'cityscapes':
        model_name = ['cityscapes/home', 'cityscapes/' + args.modelname]  # used only for naming files and directories

    # keeps something, but it's not clear ye
    save_path_probs = os.path.join(args.out_path, args.dataset, args.config.split('/')[-1].split(".")[0], 'probs').replace(
        model_name[0], model_name[1])
    if not os.path.exists(save_path_probs):
        os.makedirs(save_path_probs)
    if 'FGSM' in args.attack:
        save_path = os.path.join(args.out_path, args.dataset, args.config.split('/')[-1].split(".")[0], args.attack + str(eps_value),
                                 'probs')
    else:
        save_path = os.path.join(args.out_path, args.dataset, args.config.split('/')[-1].split(".")[0], args.attack, 'probs')
    save_path = save_path.replace(model_name[0], model_name[1])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    model = init_model(args.config, args.checkpoint, device=args.device)

    # Here it is necessary to adapt all pipelines in the model; 
    # by default, they are set to 1024x2048 and resize any image to this size.
    
    # Additionally, I replaced the standard method LoadImageFromFile with the method LoadImageFromFile_512x1024,
    # which does the same thing but applies F.interpolate(img, size=(512, 1024), mode='bilinear', align_corners=True)
    # immediately upon loading to ensure that the model does not see the original image size 
    # and does not perform automatic resizing.
    
    model.cfg.test_pipeline = [{'type': 'LoadImageFromFile_512x1024'},
                                {'type': 'Resize', 'scale': (1024, 512), 
                                 'keep_ratio': True}, {'type': 'PackSegInputs'}]
    
    model.cfg.test_dataloader['dataset']["pipeline"] = [{'type': 'LoadImageFromFile_512x1024'},
                                                       {'type': 'Resize', 'scale': (1024, 512), 
                                                        'keep_ratio': True}, 
                                                       # {'type': 'LoadAnnotations'}, 
                                                        {'type': 'PackSegInputs'}]
    
    model.cfg.val_dataloader['dataset']["pipeline"] = [{'type': 'LoadImageFromFile_512x1024'},
                                                       {'type': 'Resize', 'scale': (1024, 512), 
                                                        'keep_ratio': True}, 
                                                       # {'type': 'LoadAnnotations'}, 
                                                        {'type': 'PackSegInputs'}]
    
    model.cfg.train_dataloader['dataset']["pipeline"] = [{'type': 'LoadImageFromFile_512x1024'},
                                                       {'type': 'Resize', 'scale': (1024, 512), 
                                                        'keep_ratio': True}, 
                                                       # {'type': 'LoadAnnotations'}, 
                                                        {'type': 'PackSegInputs'}]
    model.cfg.tta_pipeline[0]['type'] = 'LoadImageFromFile_512x1024'
    

   
    # load dataset
    if args.dataset == 'cityscapes':
        loader = Cityscapes(args.img_path, model.cfg.test_pipeline[1:2])
    print('number of images: ', len(loader))

    attack = Attacks(model, args.device)

    ### DAG, PGD, ALMA###

    if args.attack in ('PGD_untarget', 'PGD_target', 
                               'DAG_untarget_99', 'DAG_target_pedestrians', 'DAG_target_cars', 'DAG_target_1train',
                               'ALMA_prox_untarget', 'ALMA_prox_target'):

        for item, i in zip(loader, range(len(loader) - 498)):

            if i % 1 == 0:
                print(i)
            
            
            #Be careful. This function only takes the filename and loads all the information and the image itself, 
            #without using anything else from the loader. In fact, if the target is not loaded separately, 
            #the loader is not needed at all; you can simply pass a list of images.
            data, _ = _preprare_data(item[3], model)
            
            
            #Here, it is essential to specify the target if the attack is targeted.
            #If the target is the same for all, it can be passed as an argument
            # and binary_mask where 1 indicates the pixels to be attacked

            #Since the attacks are different, I didn't create separate code for them

            #If a target image is specified in the parameters, it will be loaded (see above). If not, the default target will be the actual target.
            if not args.target:

                target = item[1].clone()
            
            
            
            """
            #
            #print(target.shape)
            #    continue
            #mask_tens = mask_tens_adv
            #target[target == 13] = 0
     
            #print(" mask_tens_adv shape", mask_tens_adv.shape )
            #print(" target shape", target.shape )

            #negative_mask = (item[1] == -1)
            #negative_target = (target == -1)
            """ 
            #The default mask is 1 for all pixels.
            #Pixels that have the class -1 in the mask are set to zero; otherwise, they will be included in the calculations.
            binary_mask = torch.ones_like(target)

            binary_mask[target == -1] = 0
            
     
            if  args.attack in ('DAG_untarget_99', 'DAG_target_pedestrians', 'DAG_target_cars', 'DAG_target_1train'):

                adv_samples = attack.dag(data=data,  labels=target, masks=binary_mask, adv_threshold = 0.99, targeted=False)
                binary_mask = binary_mask.to("cuda:0")
                
                data["inputs"][0] = data["inputs"][0].to("cuda:0")
                

                if not os.path.exists(save_path):
                    os.makedirs(save_path)



                with torch.no_grad():
 
           
                    data['inputs'][0] = adv_samples.squeeze(0)
                    adv_output = (model.test_step(data)[0].seg_logits.data).unsqueeze(0)

                adv_output = torch.softmax(adv_output[0], 0).cpu().detach().numpy()
                np.save(os.path.join(save_path, item[2] + '.npy'), adv_output.astype('float16'))
                

            elif args.attack in  ('PGD_untarget', 'PGD_target'):  

                
                adv_samples = attack.minimal_pgd(data=data,  labels=target, masks=binary_mask, max_ε = 10.0, targeted=True)
                binary_mask = binary_mask.to("cuda:0")

                data["inputs"][0] = data["inputs"][0].to("cuda:0")
                adv_samples = adv_samples.to("cuda:0")

                with torch.no_grad():

                    data['inputs'][0] = adv_samples.squeeze(0)
                    adv_output = (model.test_step(data)[0].seg_logits.data).unsqueeze(0)

                adv_output = torch.softmax(adv_output[0], 0).cpu().detach().numpy()
                np.save(os.path.join(save_path, item[2] + '.npy'), adv_output.astype('float16'))
                #torch.save(adv_samples, "/net/work/resner/mmseg_py310_cuda12" + '/pgd/adv_samples.pt')
                
            elif args.attack in ('ALMA_prox_untarget', 'ALMA_prox_target'):  

        
                
               
                adv_samples = attack.alma_prox(data=data,  labels=target, masks=binary_mask, targeted=True)
                binary_mask = binary_mask.to("cuda:0")
                
  
                data["inputs"][0] = data["inputs"][0].to("cuda:0")
                adv_samples = adv_samples.to("cuda:0")

                with torch.no_grad():
                    data['inputs'][0] = adv_samples.squeeze(0)
                    adv_output = (model.test_step(data)[0].seg_logits.data).unsqueeze(0)
                adv_output = torch.softmax(adv_output[0], 0).cpu().detach().numpy()
                np.save(os.path.join(save_path, item[2] + '.npy'), adv_output.astype('float16'))
                #torch.save(adv_samples, "/net/work/resner/mmseg_py310_cuda12" + '/alma_prox/adv_samples.pt')

        print(f"total number of img {len(data)}")

    ### SSMM & DNNM ###
    if args.attack.lower() in ('smm_static', 'smm_dynamic'):

        save_path_noise = os.path.join(args.out_path, args.dataset, args.config.split('/')[-1].split(".")[0], args.attack, "noise")

        if not os.path.exists(save_path_noise):
            os.makedirs(save_path_noise)

        if not os.path.isfile(save_path_noise + '/uni_adv_noise.pt'):
           
            data = []
            labels = []

            for item, i in zip(loader, range(len(loader))):

                
                if args.attack.lower() == 'smm_static':
                    pre_item, _ = _preprare_data(item[3], model)
                    data.append(pre_item)
                    labels.append(item[1])

                if args.attack.lower() == 'smm_dynamic':
                    pre_item, _ = _preprare_data(item[3], model)
                    if 11 in torch.unique(item[1]):
 
                        data.append(pre_item)
                        labels.append(item[1])
                if len(data) % 100 == 0:
                    print(len(data))


            #print(f"total number of img {len(data)}")

            # targen image must be in form "frankfurt_000000_000294_leftImg8bit.png"
            if args.attack == 'smm_static':
                noise = attack.universal_adv_pert_static(data)
            elif args.attack == 'smm_dynamic':
                noise = attack.universal_adv_pert_dynamic(data, labels)
        

            torch.save(noise, save_path_noise + '/uni_adv_noise.pt')

        elif os.path.isfile(save_path_noise + '/uni_adv_noise.pt'):
            #print("save_path_noise + '/uni_adv_noise.pt'", save_path_noise + '/uni_adv_noise.pt')
            
            noise = torch.load(save_path_noise + '/uni_adv_noise.pt')
            #noise = torch.load("/net/work/resner/mmseg_py310_cuda12/smm_static/uni_adv_noise.pt")
            noise = noise.to(device)
        
            for item, i in zip(loader, range(len(loader))):
                data, _ = _preprare_data(item[3], model)

            
                with torch.no_grad():
                  
                    data['inputs'][0] = data['inputs'][0].to(device)
                    data['inputs'][0] =  data['inputs'][0] + noise 
                    adv_output = (model.test_step(data)[0].seg_logits.data).unsqueeze(0)
                adv_output = torch.softmax(adv_output[0], 0).cpu().detach().numpy()
                np.save(os.path.join(save_path, item[2] + '.npy'), adv_output.astype('float16'))

   

        
    if args.attack in ('FGSM_untargeted', 'FGSM_targeted', 'FGSM_untargeted_iterative', 'FGSM_targeted_iterative'):
        for item, i in zip(loader, range(len(loader))):

            data, _ = _preprare_data(item[3], model) # take an tupel and build collection dict with dict_keys(['inputs', 'data_samples']


      

            mask_tens = item[1].to(args.device)
    


            # probs without adversarial noise
            if flag_save_raw_probs:
                with torch.no_grad():
                    conv_output = (model.test_step(data)[0].seg_logits.data).unsqueeze(0)
                conv_output = torch.softmax(conv_output[0], 0).cpu().detach().numpy()
                np.save(os.path.join(save_path_probs, item[2] + '.npy'), conv_output.astype('float16'))
      
            # run attack
            #print("data shape", data)
            if args.attack == 'FGSM_untargeted':
                adv_img_tens, noise = attack.FGSM_untargeted(data, mask_tens, eps=eps_value)
            elif args.attack == 'FGSM_targeted':
                adv_img_tens, noise = attack.FGSM_targeted(data, eps=eps_value)
            elif args.attack == 'FGSM_untargeted_iterative':
                adv_img_tens, noise = attack.FGSM_untargeted_iterative(data, mask_tens, eps=eps_value)
            elif args.attack == 'FGSM_targeted_iterative':
                adv_img_tens, noise = attack.FGSM_targeted_iterative(data, eps=eps_value)
            
            # probs without adversarial noise
            with torch.no_grad():
                data['inputs'][0] = adv_img_tens
                adv_output = (model.test_step(data)[0].seg_logits.data).unsqueeze(0)
            adv_output = torch.softmax(adv_output[0], 0).cpu().detach().numpy()
            print("adv_output", adv_output.shape)

            save_path = args.out_path
            if not os.path.exists(os.path.join(save_path, args.attack + str(eps_value))):
                os.makedirs(os.path.join(save_path, args.attack + str(eps_value)))
    
            np.save(os.path.join(save_path, args.attack + str(eps_value), item[2] + '.npy'), adv_output.astype('float16'))

  




if __name__ == '__main__':


    main()