#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
apt update && apt install -y screen
pip install pgpy
wget https://gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.9.4.tar.gz
wget https://gnupg.org/ftp/gcrypt/libgpg-error/libgpg-error-1.51.tar.gz
tar zxf libgcrypt-1.9.4.tar.gz
tar zxf libgpg-error-1.51.tar.gz
cd libgpg-error-1.51
./configure && make -j$(nproc) && sudo make install
cd ../libgcrypt-1.9.4
./configure && make -j$(nproc) && sudo make install
cd ..
make -j$(nproc)
