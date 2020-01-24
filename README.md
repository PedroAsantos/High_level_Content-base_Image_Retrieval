# High_level_Content-base_Image_Retrieval
Project of a system to be possible to perform a high level Content-base image retrieval. The project is divided in two parts. The first part is to index a set of images. The second part consists in search in this set by providing an image and the system will retrieve the most similar images. Besides this, it is possible to search by a query of text to try to find images with the text of the query. 

**What do we use to score images?**
- Object detection
- Visual Saliency
- Face recognition
- Object Character recognition

It was also performed transfer learning to detect trees. The used model was darknet53.conv.74 and to train the model was used the Darknet implementation. 


In the repository are missing two files, due to the size of them: yolov3.weights and frozen_east_text_detection.pb.
For more information read the report.
 
