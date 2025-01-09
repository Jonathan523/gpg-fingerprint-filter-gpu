#!/bin/bash
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
echo "Installing dependencies..."
apt update && apt install -y screen cron
pip install pgpy
echo "Downloading dependencies..."
wget -q https://gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.9.4.tar.gz 
wget -q https://gnupg.org/ftp/gcrypt/libgpg-error/libgpg-error-1.51.tar.gz
tar zxf libgcrypt-1.9.4.tar.gz
tar zxf libgpg-error-1.51.tar.gz
cd libgpg-error-1.51
echo "Building and Installing libgpg-error..."
./configure && make -j$(nproc) && make install
cd ../libgcrypt-1.9.4
echo "Building and Installing libgcrypt..."
./configure && make -j$(nproc) && make install
cd ..
echo "Building gpg-fingerprint-filter-gpu..."
make -j$(nproc)

SESSION_PREFIX="pgp_"
for ((i=0; i<${#TASKS[@]}; i++)); do
    SESSION_NAME="${SESSION_PREFIX}${i}"
    TASK="./gpg-fingerprint-filter-gpu -j 34 -a ed25519 -t 59654321 -m Y \"x{10}|x{11}|x{12}|x{13}|x{14}|x{15}|x{16}|x{4}y{8}\" /root/fs/output.pgp"
    screen -dmS "$SESSION_NAME"
    screen -X stuff "$TASK\n" "$SESSION_NAME"
    echo "Started task $TASK in screen session $SESSION_NAME"
done

CRON_ENTRY="* * * * * python $(pwd)/ca.py /root/fs/output.pgp/ /root/fs/dist/ >> /root/fs/log.txt"
echo "$CRON_ENTRY" | crontab -
echo "Added cron entry."