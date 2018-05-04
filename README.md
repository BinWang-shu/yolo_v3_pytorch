# yolo_v3_pytorch
An simple implement of yolo_v3 on pytorch

This is the simple implement of the yolov3[https://pjreddie.com/media/files/papers/YOLOv3.pdf].<br>
[https://github.com/ayooshkathuria/pytorch-yolo-v3.git] and [https://github.com/longcw/yolo2-pytorch.git] is referred.

I have finish the train and test of the implement but I can not ensure the accuracy.<br>
The loss convergence in the training.

Bacause I create the net in myself without the offical cfg, I do not design the code to copy the weights from pre_trained model.<br>
If you want to use the pre_trained model to detect, you can find some implemments from other authors in github.

## TODO
* finsih the evalution of the VOC
* achieve the results as the offical results.



