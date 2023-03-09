git clone https://github.com/lllyasviel/ControlNet.git
ln -s ControlNet/ldm ./ldm
ln -s ControlNet/cldm ./cldm
ln -s ControlNet/annotator ./annotator
cd ControlNet/models
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_hed.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_mlsd.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_normal.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_scribble.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_seg.pth
cd ../../
