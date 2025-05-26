# Start Guide for UG coursework
#### Kian Pflitsch 
1. Download coursework dataset zip file
2. Place downloaded zip file into CW_Dataset
3. Extract contents in the same directory.  
The file structure should look like the following:

```
   CW_Dataset/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── train/
        ├── images/
        └── labels/
```
4. Run Visualise_Normalize_Dataset.ipnyb and a new directory will be created called base_dataset
5. 


```
   Code/
    ├── main.ipynb
    └── Visualise.ipynb
   CW_Dataset/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── train/
        ├── images/
        └── labels/
   dataset_as_pkl/ (code will automatically output to this file)
    └── # features stored from running feature extraction 
   Models/
    └── openCV_Face_Detection/
        ├── deploy.prototxt.txt
        └── res10_300x300_ssd_iter_140000.caffemodel
   Personal_Dataset/
   └── # your own images (jpg, png)
   test_function.ipynb

```
