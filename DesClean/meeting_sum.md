
*hyperparameter tuning:

*parameters:

learning rate: 0.00002,0.0002, 0.01(default) 0.00002 would have the best convergence result. while for some cases, the rest would lead to a fast but non-sense result.

training samples: speed testing: use 8 samples(1 batch) trained for 400 times(20 step of epoch, 20 epoch) and produces the result for the sample, this would provide a outlook for the performence of the network, mainly whether would converge. normal testing, use part of the complete training set, all trained for 20 epochs, steps vary from 200-2000. 
For most of the case, the speed test would have same training pattern as normal test. if the network cannot converge with speed test, it cannot pass the normal test as well

loss: compare to randering loss, l1 loss has better convergences. The randering loss would lead to divereging, which need more experiment about the reasoning, but not at the moment. 

dimension of filters: Tried same dimension as presented in the paper(128,256,512,512,512,512,512,512), 8 layers, 6 512 filters. also tried (32,64,128,256,512,512,512,512). This did not affect the result significantly. Eaxct layers of unet: (64,128,256,512,1024).

convolution layers: the example implementation of unet is for each stage, there are two layers of convolutions. I tried the one layers. the single layered version would have better convergence with small data set and speed. this is just the under fitting of the over complicated network. In conclusion, this would not affect the convergence of the network.

activation function sigmoid works better than softmax even with [0,1] range. and for the activation before convolution, selu and relu, would have minor diffierence. 

input range: I tried convert the image from 0-1 to -1-1 by multiplying 2 and -1. However, the network wont converge for this range, not sure the reason. 

############
best mse is around 0.0018, but by seeing the prediction, it only predict the overall mean colour. The details are not preserved.





current mse:
current accuracy: 0.63
current loss: 0.006

bright square top left? not sure why(or does it exist in the prediction)

next steps: 
1. first try to use the hdr images
   - should not be much of trouble with the hdr training data
2. comparing with li and boss's network
   - 
3. building with 3D inputs
   - 






