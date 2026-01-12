import os
import torch
import argparse
import numpy as np
import torch.nn as nn

import matplotlib as mpl
mpl.use('Agg')
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.utils import get_logger
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

from BMPWriter import BMPWriter as bmpw
import openslide
tilecnt = 0

parser = argparse.ArgumentParser(description="Params")
parser.add_argument( "--model_path", nargs="?", type=str, default="runs/DMMN_breast_model.pkl")
parser.add_argument( "--out_path", nargs="?", type=str, default="imgs", help="Path of the output segmap")
args = parser.parse_args()

### TNBC segmentation classes ###
# Tumor: red
# Stroma: blue
# Inflammatory infiltration: green
# Necrosis: yellow
# Other/background: gray

def test(test_file, data_path):
    bmp = bmpw()
    model_file_name = os.path.split(args.model_path)[1]
    model_name  = {}
    model_name['arch'] = "DMMN"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = 6 # the number of tissue subtype classes + 1
    
    with open(test_file) as f1:
        test_file_patches = [line.rstrip('\n') for line in f1]
    
    tile_size = 1024*2
    problem_type = 'tissue'
    batch_size = 1
    with open(test_file) as f:
        file_names = [line.rstrip('\n') for line in f]
    
    model = get_model(model_name, n_classes)
    state = convert_state_dict(torch.load(args.model_path, weights_only=False)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.cuda()

    if not os.path.exists(str(args.out_path)):
        os.makedirs(str(args.out_path))

    test_file_patches_split = test_file_patches[0].split(",")    
    pslide_id = int(test_file_patches_split[0])    
    filename = os.path.join(os.path.abspath(args.out_path) + "/" + str(pslide_id) + ".svs_data/predictions.bmp")    
    slide = openslide.OpenSlide(data_path + str(pslide_id) + ".svs")
    tilecnt = 0
    image_size = slide.dimensions
    outfile = bmp.makeempty(int(image_size[0]/2), int(image_size[1]/2))
    with torch.no_grad():
        for ii in tqdm(range(0,len(test_file_patches))):
            test_file_patches_split = test_file_patches[ii].split(",")
            slide_id = int(test_file_patches_split[0])
            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            inputs_slide = slide.read_region((xmin,ymin), 0, (tile_size,tile_size)).convert('RGB')
            inputs_slide = np.array(inputs_slide)/255.0
            inputs_slide = np.expand_dims(np.transpose(inputs_slide, (2,1,0)), axis=0)
            inputs = torch.from_numpy(inputs_slide).float().to(device)
            inputs = torch.flip(inputs.permute(0,1,3,2),[1])
            
            if tilecnt >= len(test_file_patches):
                outfile = color_change(outfile)
                image_size = slide.dimensions
                bmp.writebmp(filename,outfile,int(image_size[0]/2), int(image_size[1]/2),palette='standard')
                break

            filename = os.path.join(os.path.abspath(args.out_path) + "/" + str(slide_id) + ".svs_data/predictions.bmp")
            if slide_id != pslide_id:
                pfilename = os.path.join(os.path.abspath(args.out_path) + "/" + str(pslide_id) + ".svs_data/predictions.bmp")
                outfile = color_change(outfile)
                image_size = slide.dimensions
                bmp.writebmp(pfilename,outfile,int(image_size[0]/2), int(image_size[1]/2),palette='standard')
                slide = openslide.OpenSlide(data_path + str(slide_id) + ".svs")
            if not os.path.isdir(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image_size = slide.dimensions
                outfile = bmp.makeempty(int(image_size[0]/2), int(image_size[1]/2))

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            if xmin == 0 and ymin == 0:
                # tile at outfile[0:128,0:128]
                ref_pad = nn.ReflectionPad2d(512*2)
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[:128,:64] = t_mask_shrink[128:,64:]

                # tile at outfile[0:128,128:384]
                ref_pad = nn.ReflectionPad2d((512*2,512*2,256*2,256*2))
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[:128,64:192] = t_mask_shrink[128:,:]

                # tile at outfile[128:384,0:128]
                ref_pad = nn.ReflectionPad2d((256*2,256*2,512*2,512*2))
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[128:384,:64] = t_mask_shrink[:,64:]

                # tile at outfile[128:384,128:384]
                ref_pad = nn.ReflectionPad2d(256*2)
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                outfile[128:384,64:192] = t_mask_shrink

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            if xmin == 0:
                ymin += 384*2
                # tile at outfile[0:128,ymin:ymin+256]
                ref_pad = nn.ReflectionPad2d((512*2,512*2,0,0))
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t2 = int(ymin/2)
                wh2 = int(image_size[1]/2)
                if t2 < wh2:
                    if t2 + 256 > wh2:
                        outfile[t2:,:64] = t_mask_shrink[0:wh2-t2,64:]
                    else:
                        outfile[t2:t2+256,:64] = t_mask_shrink[:,64:]

                # tile at outfile[128:384,ymin:ymin+256]
                ref_pad = nn.ReflectionPad2d((256*2,256*2,0,0))
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t2 = int(ymin/2)
                wh2 = int(image_size[1]/2)
                if t2 < wh2:
                    if t2 + 256 > wh2:
                        outfile[t2:,64:192] = t_mask_shrink[0:wh2-t2,:]
                    else:
                        outfile[t2:t2+256,64:192] = t_mask_shrink

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            if ymin == 0:
                xmin += 384*2
                # tile at outfile[xmin:xmin+256,0:128]
                ref_pad = nn.ReflectionPad2d((0,0,512*2,512*2))
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t1 = bmp.getrowsize(int(xmin/2))
                wh1 = bmp.getrowsize(int(image_size[0]/2))
                if t1 < wh1:
                    if t1 + 128 > wh1:
                        outfile[:128,t1:] = t_mask_shrink[128:,0:wh1-t1]
                    else:
                        outfile[:128,t1:t1+128] = t_mask_shrink[128:,:]

                # tile at outfile[xmin:xmin+256,128:384]
                ref_pad = nn.ReflectionPad2d((0,0,256*2,256*2))
                inputs_new = torch.zeros([1,3,1024*2,1024*2]).cuda()
                inputs_new = ref_pad(inputs)[:,:,:1024*2,:1024*2]
                outputs = model(inputs_new[:,:,::2,::2][:,:,384:640,384:640],inputs_new[:,:,::4,::4][:,:,128:384,128:384],inputs_new[:,:,::8,::8])
                t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
                t_mask = t_masks[0,:,:]
                t_mask_shrink = mask_shrink(t_mask)
                t1 = bmp.getrowsize(int(xmin/2))
                wh1 = bmp.getrowsize(int(image_size[0]/2))
                if t1 < wh1:
                    if t1 + 128 > wh1:
                        outfile[128:384,t1:] = t_mask_shrink[:,0:wh1-t1]
                    else:
                        outfile[128:384,t1:t1+128] = t_mask_shrink

            xmin = int(test_file_patches_split[1])
            ymin = int(test_file_patches_split[2])
            outputs = model(inputs[:,:,::2,::2][:,:,384:640,384:640],inputs[:,:,::4,::4][:,:,128:384,128:384],inputs[:,:,::8,::8])
            t_masks = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
            t_mask = t_masks[0,:,:]
            xmin += 384*2
            ymin += 384*2
            t_mask_shrink = mask_shrink(t_mask)
            t1 = bmp.getrowsize(int(xmin/2))
            t2 = int(ymin/2)
            image_size = slide.dimensions
            wh1 = bmp.getrowsize(int(image_size[0]/2))
            wh2 = int(image_size[1]/2)
            if t1 < wh1 and t2 < wh2:
                if t2 + 256> wh2 and t1 + 128 > wh1:
                    print(slide_id)
                elif t2 + 256 > wh2:
                    outfile[t2:,t1:t1+128] = t_mask_shrink[0:wh2-t2,:]
                elif t1 + 128 > wh1:
                    outfile[t2:t2+256,t1:] = t_mask_shrink[:,0:wh1-t1]
                else:
                    outfile[t2:t2+256,t1:t1+128] = t_mask_shrink

            pslide_id = slide_id
            tilecnt += 1
    outfile = color_change(outfile)
    image_size = slide.dimensions
    bmp.writebmp(filename,outfile,int(image_size[0]/2), int(image_size[1]/2),palette='standard')

def color_change(t_mask):    
    t_mask[t_mask == 0] = 5*17    
    return t_mask

def mask_shrink(t_mask):
    # for color-coding      
    t_mask_shrink = t_mask[:,::2]*17
    return t_mask_shrink


if __name__ == "__main__":

    test_file = "test_coords_S01.csv"       # the list of patch coordinates
    data_path = "testing_images/"           # the path where testing whole slide images are located
    test(test_file, data_path)

        
        
