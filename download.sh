#!/bin/bash

git clone https://github.com/lllyasviel/ControlNet.git
ln -s ControlNet/ldm ./ldm
ln -s ControlNet/cldm ./cldm
ln -s ControlNet/annotator ./annotator
cd ControlNet/models

function download_model() {
    model=$1
    path=$model
    url="https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/$model"
    md5sum_expected=$2
    if [ -f $path ]; then
        md5sum_actual=$(md5sum $path | awk '{print $1}')
        if [ "$md5sum_actual" == "$md5sum_expected" ]; then
            echo "$model is already downloaded and has the expected MD5 checksum."
            return
        fi
    fi
    echo "Downloading $model ..."
    wget $url 
    md5sum_actual=$(md5sum $path | awk '{print $1}')
    if [ "$md5sum_actual" != "$md5sum_expected" ]; then
        echo "Error: $model download failed or MD5 checksum does not match."
        exit 1
    fi
}

download_model control_sd15_canny.pth  680f1938ae2116941c7b95165ae0b293
download_model control_sd15_depth.pth  6c1c59867120a4fdfc009fb7c81a9fb6
download_model control_sd15_hed.pth  aa3d5675393ecfa8d614fb1afa13c0af
download_model control_sd15_mlsd.pth  912ff3d292e496cc6e167ef4edd6b5cf
download_model control_sd15_normal.pth  599b5e8d30619150488be33f1889aaa8
download_model control_sd15_openpose.pth  6f7a19a1c066889e51cff677c14f7a51
download_model control_sd15_scribble.pth  276e199415b23d23d4a7454730422e77
download_model control_sd15_seg.pth  dadb5c06ad67dc9ef18a1f63c86c897e

cd ../../
