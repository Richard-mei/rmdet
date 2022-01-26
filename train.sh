echo " Start to train..."

echo " training r50-bifpn.yaml"
python train.py --config configs/r50-bifpn.yaml

echo ""
echo ""
echo " training rep-bifpn-gs-gf"
python train.py --config configs/rep-bifpn-gs-gf.yaml

echo ""
echo ""
echo " training rep-fpn-gs-gf"
python train.py --config configs/rep-fpn-gs-gf.yaml

