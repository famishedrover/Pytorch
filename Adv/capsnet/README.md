## CAPSULE NETS

### References :
[Link To Paper ](https://arxiv.org/abs/1710.09829)
[Blog](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc)
[Keras+Explanation](https://www.analyticsvidhya.com/blog/2018/04/essentials-of-deep-learning-getting-to-know-capsulenets/)
### Requirements :
1. PyTorch & TorchVision
2. Cuda 9.0 (if on GPU)
3. Py 2 (Untested on Py3)
4. Preferable make and run a virtual environment for Python.

### Running The scripts :
1. `screen -L -S train` on remote server.
2. Run `python capsnet.py`
3. Run `python plotcapsnet.py -d` to download and plot or `python plotcapsnet.py` to plot using locally sotred `screenlog.0`
4. View screenlogs @ `screenlogs.0`

#### Making Changes to `plotcapsnet.py`
Change `server` , `serverDir` to point to your remote gpu.
I have used AWS p2.xlarge instance in this case.

### Hyperparameters
* num_epochs = 10
* learning_rate = 0.01
* batch_size = 128
* test_batch_size = 128
* early_stop_loss = 0.0001

#### Other Parameters
* conv_inputs=1
* conv_outputs=256
* num_primary_units=8
* primary_unit_size=32*6*6
* output_unit_size=16

### Reconstruction Images
![recontruction_20](https://user-images.githubusercontent.com/18165020/38762022-413bdd22-3fa5-11e8-8737-b3508f7d8b7f.png)
![recontruction_750](https://user-images.githubusercontent.com/18165020/38762024-47d76ef8-3fa5-11e8-8209-45710bce5c17.png)
![recontruction_1580](https://user-images.githubusercontent.com/18165020/38762026-4bb4ccaa-3fa5-11e8-92e7-4ca4a554f0e0.png)
![recontruction_2140](https://user-images.githubusercontent.com/18165020/38762027-4ec1bfd4-3fa5-11e8-8314-849f58ae261d.png)
![recontruction_2860](https://user-images.githubusercontent.com/18165020/38762029-51573d78-3fa5-11e8-85d0-b64e5935c01f.png)
![recontruction_3520](https://user-images.githubusercontent.com/18165020/38762033-56bfd27a-3fa5-11e8-99d4-4d77ca79f285.png)


### Plots
![plots](https://user-images.githubusercontent.com/18165020/38762049-885a6b2e-3fa5-11e8-94d5-66f48ed426c6.png)

* Loss : 100 values per epoch
* Accuracy : per epoch
* Time Taken in Minutes per epoch
