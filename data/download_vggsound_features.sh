#!/bin/bash

if (! command -v tar &> /dev/null) | (! command -v curl &> /dev/null) | (! command -v md5sum &> /dev/null)
then
    echo "tar or curl or md5sum commands not found"
    echo ""
    echo "Please, install it first."
    echo "If you cannot/dontwantto install these, you may download the features manually."
    echo "You may find the links and correct paths in the repo README."
    echo "Make sure to check the md5 sums after manual download."
    echo "Extraction commands can be checked in this file."
    exit
fi

download_check_expand_rmtar () {
    # $1: $BASE_LINK
    # $2: $FNAME
    # $3: $WHERE_TO
    # $4: $MD5SUM_GT
    echo "Downloading" $2
    curl $1/$2 --create-dirs -o $3/$2
    echo "Checking tar md5sum"
    cat $4 | grep $2 | md5sum --check
    echo "Expanding tar"
    tar xf $3/$2 -C $3
    echo "Removing tar"
    rm $3/$2
    echo ""
}

WHERE_TO="./downloaded_features"
BASE_LINK="https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggsound"
MD5SUM_GT="./md5sum_vggsound.md5"

for i in {01..64}; do
    echo "Starting" $i "/ 64 part"

    # spectrograms
    FNAME="melspec_10s_22050hz_$i.tar"
    download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT

    # BN Inception Features
    FNAME="feature_rgb_bninception_dim1024_21.5fps_$i.tar"
    download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT

    FNAME="feature_flow_bninception_dim1024_21.5fps_$i.tar"
    download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT

    # ResNet50 Features
    # FNAME="feature_resnet50_dim2048_21.5fps_$i.tar"
    # download_check_expand_rmtar $BASE_LINK $FNAME $WHERE_TO $MD5SUM_GT
done
