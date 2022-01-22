echo "****************** Installing pytorch 1.8.0 ******************"
conda install -y pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
pip install matplotlib

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools


echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX tensorboard


echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

echo ""
echo ""
echo "****************** Installing colorlog ******************"
pip install colorlog


echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing cupy ******************"
pip install cupy-cuda111

echo ""
echo ""
echo "****************** Installing requests ******************"
pip install requests

echo ""
echo ""
echo "****************** Installing onnx ******************"
pip install onnx

echo ""
echo ""
echo "****************** Installing imgaug ******************"
pip install imgaug

echo ""
echo ""
echo "****************** Installing seaborn ******************"
pip install seaborn

echo ""
echo ""
echo "****************** Installing yacs ******************"
pip install yacs

echo ""
echo ""
echo "****************** Installation complete! ******************"
