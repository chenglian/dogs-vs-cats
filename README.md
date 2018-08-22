# dogs-vs-cats
Case Studies: Dogs vs Cats

Kaggle Deep Learning competition: Write an algorithm to classify whether images contain either a dog or a cat. 

- The challenge was published on Kaggle in 2013, with varying image resolutions.

- The training data contains 25,000 images of dogs and cats. There is only a small subset of the images uploaded to this repo.

- https://www.kaggle.com/c/dogs-vs-cats/data

This exercise is more for a self-learning purpose.

# Software stack

- Keras

- Tensorflow

- cuDNN (cuDNN(CUDA Deep Neural Network library - a GPU-accelerated library of primitives for neural networks, it provides highly tuned implementations for: convolution,
pooling, normalization, and activation layers.)

- CUDA (Compute Unified Device Architecture - a parallel programming model that enables dramatic increases in computing performance by harnessing GPU)

- GPU

# Setup GPU environment for deep learning exercises

- Create a p2.xlarge Ubuntu 16.04 VM Instance with at least 20G storage in US East (N. Virginia) region (not all AWS regions offer GPU). You will need to request and pay for this type of VMs.

- Install CUDA Toolkit and cuDNN library
  
  ```
  $ wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
  $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
  $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
  $ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
  
  $ sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
  $ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
  
  $ sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
  $ sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
  $ sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
  $ sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
  
  $ sudo apt-get update
  $ sudo apt-get install cuda=9.0.176-1
  $ sudo apt-get install libcudnn7-dev
  $ sudo apt-get install libnccl-dev
  
  ### Reboot the system to load the NVIDIA drivers.
  $ sudo reboot
  ```
  
- Set environment variables in .bashrc file
  
  ```
  $ echo 'export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
  $ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  $ source ~/.bashrc
  ```
  
- Install Python environment and libraries
  
  ```
  $ wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
  $ bash Anaconda3-5.2.0-Linux-x86_64.sh -b -p ~/anaconda
  $ echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc
  $ source ~/.bashrc
  $ conda update conda
  $ conda create -n py36-venv python=3.6 anaconda
  $ source activate py36-venv
  ```
  
- Install TensorFlow (GPU-accelerated version) and Keras
  
  ```
  $ pip install tensorflow-gpu==1.5.0
  $ pip install keras
  ```
  
- Install Graphviz and Pydot to visualize neural network
 
  ```
  $ sudo apt-get install graphvi
  $ pip install pydot
  ```
 
- Remotely access to Jupyter Notebook host in AWS
 
  ```
  $ source activate py36-venv
  $ nohup jupyter notebook --no-browser
  $ nohup ssh -i ~/path/aws_key.pem -L 8080:localhost:8888 ubuntu@aws_public_ip
  Copy and Paste below URL from nohup.out into your local browser. http://localhost:8080/?token=your_jupyter_token
  ```
 
- Test Tensorflow using GPU in Jupyter Notebook

  ```
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
  ```
  
  ```
    /home/ubuntu/anaconda/envs/py36-venv/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
    from ._conv import register_converters as _register_converters
  ```
  
  ```
    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 15064103543287563568
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 174784512
    locality {
      bus_id: 1
    }
    incarnation: 14824778902671040536
    physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0, compute capability: 3.7"
    ]
  ```

