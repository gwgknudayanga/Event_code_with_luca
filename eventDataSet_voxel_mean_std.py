import os
import random
import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tr
import cv2
import h5py

import numpy as np

#from numpy.lib.recfunctions import structured_to_unstructured


class ClassificationDataset(Dataset):
    
    def __init__(self,mode,data_folder_path,annotationFile,inputRepresentationType,image_shape,dataSet_name,numOfTimeSteps = 5,sampleSize = 97620,numOfTbins = 2,transform = None):
         #whether test or train
        #Representation type
        #type = 0 raw images
        #type = 1 event cube 
        #type = 2 voxel grid
        #type = 3 meanStd 6 channel input
        
        self.root = data_folder_path
        self.dataSet_name = dataSet_name
        self.tbin = numOfTbins
        self.T = numOfTimeSteps
        self.C = self.tbin * 2
        self.sample_size = sampleSize
        self.quantization_size = [self.sample_size//self.T,1,1] #sample length for one time step
        self.w, self.h = image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]
        self.mode = mode
        self.inputRepresentationType = inputRepresentationType # representation type = 0 means event cube, 1 means meanStd and 2 means voxel cube
        self.img_labels = pd.read_csv(annotationFile)
        print("Number of ",mode," events ",len(self.img_labels))
        self.transform = transform
        
    def __getitem__(self, index):
        
        event_file_path = self.img_labels.iloc[index, 0]
        if not event_file_path.startswith("/work3"):
            event_file_path = os.path.join(self.root,self.mode,self.img_labels.iloc[index, 0])

        label = self.img_labels.iloc[index,1]

        #print("event file path is ",event_file_path)
        events = load_v2e_event_data(event_file_path)

        if(events.size == 0):
            print("Empty sample ....", event_file_path, "for class")
            return None

        if (len(events[events[:,3] > 1])):
            print("large event value found ",event_file_path)

        if self.inputRepresentationType == 0: #event cube representation for SNN
            eventcube = self.getEventCubeRepresentation(events)
            if self.transform:
                eventcube = self.transform(eventcube)
            return eventcube,label

        elif self.inputRepresentationType == 1: #Voxel grid reresentation

            voxel_grid_bins = self.T * self.tbin#* self.tbin
            voxel_grid = self.prepare_and_return_voxel_grid(events,voxel_grid_bins, self.w, self.h)  # shape of the vocel grid is (self.T,height,width)
            if self.transform:
                voxel_grid = self.transform(voxel_grid)
            return voxel_grid,label

        elif self.inputRepresentationType == 2:
            histMeanStdMap = self.getHistMeanStdFMaps(events,self.w,self.h) # this map is (6,height,width) dimension; Need to do the input 
                                                                                    #normalization with input batch normalization
            return histMeanStdMap,label
    

        """coords, feats, target = self.samples[index]
        sample = torch.sparse_coo_tensor(coords.t(), feats.to(torch.float32)).coalesce()
        sample = sample.sparse_resize_(
            (self.T, sample.size(1), sample.size(2), self.C), 3, 1
        ).to_dense().permute(0,3,1,2)                     #(T,C,W,H)
        
        sample = Tr.Resize((227,227), Tr.InterpolationMode.BILINEAR)(sample)
        return sample, target"""

    def __len__(self):
        #return len(self.samples)
        return len(self.img_labels)

    def build_dataset(self, data_dir, save_file):
        raise NotImplementedError("The method build_dataset has not been implemented.")
    

def load_v2e_event_data(filename):
    """Load V2E Events, all HDF5 records."""

    assert os.path.isfile(filename)

    v2e_data = h5py.File(filename, "r")
    events = v2e_data["events"][()]
    events_1 = events[events[:,0] < 50000]

    return events_1


class v2EDamageDataSet(ClassificationDataset):
    def __init__(self,mode,data_folder_path,annotationFile,inputRepresentationType,image_shape = (346,260),dataSet_name = "v",numOfTimeSteps = 5,sampleSize = 97620,numOfTbins=2,transform = None):
        super().__init__(mode,data_folder_path,annotationFile,inputRepresentationType,image_shape,dataSet_name,numOfTimeSteps,sampleSize,numOfTbins,transform)
        self.voxelStdLast = 0
    
    """def build_dataset(self,data_dir,save_file):
        print("building v2e damage dataset ....")
        
        classIDvsClassNameDict = {}
        classes_dir = [os.path.join(data_dir,class_name) for class_name in os.listdir(data_dir)]
        samples = []
        count = 0
        count2 = 0
        
        current_max = 0
        print("class dir ",classes_dir)
        for class_id,class_dir in enumerate(classes_dir):
            self.files = [os.path.join(class_dir,time_seq_name) for time_seq_name in os.listdir(class_dir)]
            target = class_id
            print(f'Building the class id {class_id} and class dir {class_dir}')
            idForAugment = 0
            if "spalling" in class_dir:
                idForAugment = 0
            elif "healthy" in class_dir:
                idForAugment = 1
            elif "crack" in class_dir:
                idForAugment = 2
            elif "corrosion" in class_dir:
                idForAugment = 3
            elif "efflorescence" in class_dir:
                idForAugment = 4            

            for file_name in self.files:
                #print("processing ",file_name)
                doAugment = True
                if self.mode != "train":
                    doAugment = False
                if idForAugment == 2:
                    count += 1
                if idForAugment == 3:
                    count2 += 1

                event_data_list = load_v2e_event_data(file_name,idForAugment,doAugment,count,count2)

                for events in event_data_list:
                    
                    #print("printing events ...",events)
                    #print("events shape ",events.shape)
                    #print("number of events",events.size)
                    if(events.size == 0):
                        print("Empty sample ....", file_name, "for class",class_id)
                        continue
                    
                    if (len(events[events[:,3] > 1])):
                        print("large event value found ",file_name)
                    # Bin the events on T timesteps
                    coords = events[:,0:3]
                    #print("coords 1 0 ",coords)
                    coords = torch.floor(coords/torch.tensor(self.quantization_size)) 
                    coords[:, 1].clamp_(min=0, max=self.quantized_w-1)
                    coords[:, 2].clamp_(min=0, max=self.quantized_h-1)
                    #print("coords 2 = ",coords)
                    if (coords[:,0].max() > 4):
                        print("exceeding time bins to ",coords[:,0].max()," ",file_name)
                    # TBIN computations
                    #else:
                    #    print("max time ",coords[:,0].max())
                    tbin_size = self.quantization_size[0] / self.tbin
                    tbin_coords = (events[:,0] % self.quantization_size[0]) // tbin_size
                    tbin_feats = (2* tbin_coords) + events[:,3]
                    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*self.tbin).to(bool)
                    
                    samples.append([coords.to(torch.int16), feats.to(bool), target])
        return samples"""
    

    def prepare_and_return_voxel_grid(self,events,numOfVoxelBins,width,height):
        
        voxel = self.events_to_voxel_grid(events,numOfVoxelBins,width, height)
        # normalization
        nonzero_ev = (voxel != 0)
        num_nonzeros = nonzero_ev.sum()

        # has events  # apply normalization
        if num_nonzeros > 0:
            mean = voxel.sum()/num_nonzeros
            stddev = np.sqrt((voxel**2).sum()/num_nonzeros-mean**2)
            print("stddev value is ",stddev)
            mask = nonzero_ev.astype("float32")
            if (stddev > 0 ):
                voxel = mask*(voxel-mean)/(stddev + 0.0001)
                self.voxelStdLast = stddev
            else:
                voxel = mask*(voxel-mean)/self.voxelStdLast

        
        voxel = torch.tensor(voxel, dtype=torch.float32)
        voxel = Tr.Resize((227,227), Tr.InterpolationMode.BILINEAR)(voxel)

        """if self.augmentation:
            # data augmentation
            if self.is_train:
                voxel = self.hflip(voxel)
                #  voxel = self.rand_rotate(voxel)
                voxel = self.rand_affine(voxel)"""
        return voxel

    
    def events_to_voxel_grid(self,events, num_bins, width, height):
        """
        Build a voxel grid with bilinear interpolation in the
        time domain from a set of events.

        WARNING! TIME IS IN SECONDS!

        :param events: a [N x 4] NumPy array containing one event per
        row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """

        assert(events.shape[1] == 4)
        assert(num_bins > 0)
        assert(width > 0)
        assert(height > 0)

        
        print("hhh ",events[events[:,3] > 1])

        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].astype(np.int)
        ys = events[:, 2].astype(np.int)
        pols = events[:, 3].astype(np.int)
        
        print(pols)
        pols[pols == 0] = -1  # polarity should be +1 / -1
        print(pols)
        tis = ts.astype(np.int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts
        #print("sssssss ",tis," ",dts," ",vals_left," ",vals_right,x,y)
        #print(pols)

        valid_indices = tis < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
                + tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        np.add.at(
            voxel_grid,
            xs[valid_indices] + ys[valid_indices] * width +
            (tis[valid_indices] + 1) * width * height,
            vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

        return voxel_grid

    def getHistMeanStdFMaps(self,events,width,height):

        print("large value for event polarity ",events[events[:,3] > 1])
        """events[events[:,3] > 1][:,3] = 1
        fullHist = np.zeros((2,height,width), np.float32)
        avgMap = np.zeros((2,height,width), np.float32)
        stdMap = np.zeros((2,height,width), np.float32)
        for x in range(width):
            temp = events[events[:,1] == x]
            for y in range(height):
                x_y_bin_events = temp[temp[:,2] == y]
                x_y_bin_events_on = x_y_bin_events[x_y_bin_events[:,3] == 1]
                x_y_bin_events_off = x_y_bin_events[x_y_bin_events[:,3] == 0]
                x_y_bin_events_on_t_stamps = x_y_bin_events_on[:,0]
                x_y_bin_events_off_t_stamps = x_y_bin_events_off[:,0]
                
                #print("rrrrr ",len(x_y_bin_events_on_t_stamps))
                #print("qqqqq ",len(x_y_bin_events_off_t_stamps))
                #if (len(x_y_bin_events_on_t_stamps) > 20 or len(x_y_bin_events_off_t_stamps) > 20):
                #    print("large number of events",len(x_y_bin_events_on_t_stamps), " ",len(x_y_bin_events_off_t_stamps))

                fullHist[0,y,x] = len(x_y_bin_events_on_t_stamps)

                fullHist[1,y,x] = len(x_y_bin_events_off_t_stamps)

                #if len(x_y_bin_events_off_t_stamps) < 1:
                #    print("found empty events at ",x,y)
                
                if  len(x_y_bin_events_on) > 0:
                    
                    avgMap[0,y,x] = np.sum(x_y_bin_events_on_t_stamps)/fullHist[0,y,x]
                    stdMap[0,y,x] = np.sqrt(np.sum(np.square(x_y_bin_events_on_t_stamps - avgMap[0,y,x]))/fullHist[0,y,x])

                if  len(x_y_bin_events_off) > 0:
                    
                    avgMap[1,y,x] = np.sum(x_y_bin_events_off_t_stamps)/fullHist[1,y,x]
                    stdMap[1,y,x] = np.sqrt(np.sum(np.square(x_y_bin_events_off_t_stamps - avgMap[1,y,x]))/fullHist[1,y,x])"""
        
        avgMap_pos = np.zeros((height, width), np.float32).ravel()
        avgMap_neg = np.zeros((height,width),np.float32).ravel()

        pos_events_arr = events[events[:,3] == 1]
        neg_events_arr = events[events[:,3] == 0]
        
                
        
        fmap = np.concatenate((fullHist,avgMap,stdMap),axis=0)
        fmap = torch.tensor(fmap, dtype=torch.float32)
        return fmap


    def getEventCubeRepresentation(self,events):
        
        # Bin the events on T timesteps
        coords = events[:,0:3]
        #print("coords 1 0 ",coords)
        coords = torch.floor(coords/torch.tensor(self.quantization_size)) 
        coords[:, 1].clamp_(min=0, max=self.quantized_w-1)
        coords[:, 2].clamp_(min=0, max=self.quantized_h-1)
        #print("coords 2 = ",coords)
        if (coords[:,0].max() > 4):
            print("exceeding time bins to ",coords[:,0].max())
        # TBIN computations
        #else:
        #    print("max time ",coords[:,0].max())
        tbin_size = self.quantization_size[0] / self.tbin
        tbin_coords = (events[:,0] % self.quantization_size[0]) // tbin_size
        tbin_feats = (2* tbin_coords) + events[:,3]
        feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*self.tbin).to(bool)

        coords, feats = [coords.to(torch.int16), feats.to(bool)]
        sample = torch.sparse_coo_tensor(coords.t(), feats.to(torch.float32)).coalesce()
        sample = sample.sparse_resize_(
            (self.T, sample.size(1), sample.size(2), self.C), 3, 1
        ).to_dense().permute(0,3,1,2)                      #(T,C,W,H)

        sample = Tr.Resize((128,128), Tr.InterpolationMode.NEAREST)(sample)
        
        return sample

def collate_fn(self, batch):     # This collate function handle empty samples returns to the dataloader by by CustomDataset
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


"""if __name__ == '__main__':
self,mode,data_folder_path,image_shape = (346,260),dataSet_name = "v",numOfTimeSteps = 5,sampleSize = 97620,numOfTbins=2):

    v2EDamageDataSet("train",folder_path_for_event_data,image_shape =(346,260),dataSet_name = "v",numOfTimeSteps = 5,sampleSize = 50000,numOfTbins = 2)
    dataloader = DataLoader(dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=os.cpu_count() - 1, 
    pin_memory=True,
    collate_fn=collate_fn)"""
