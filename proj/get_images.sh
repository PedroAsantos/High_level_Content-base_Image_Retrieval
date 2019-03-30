#!/usr/bin/env bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mkdir -p test
cd val2017
# last 10 pictures used for testing purposes
ls -t | tail -n 10 | xargs -I {} mv {} ../test
# others are removed
ls -t  | tail -n 4490 | xargs rm
# we are left with 500 images, they will the album we'll index
mv val2017 album
