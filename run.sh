#!/bin/bash
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "export PATH=/usr/local/cuda/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
source /etc/ai_booster enable
echo "Installing dependencies..."
DEBIAN_FRONTEND=noninteractive apt update -qq && apt install -y screen -qq
pip install pgpy -qq
echo "Downloading dependencies..."
wget -q https://gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.9.4.tar.gz 
wget -q https://gnupg.org/ftp/gcrypt/libgpg-error/libgpg-error-1.51.tar.gz
tar zxf libgcrypt-1.9.4.tar.gz
tar zxf libgpg-error-1.51.tar.gz
cd libgpg-error-1.51
echo "Building and Installing libgpg-error..."
./configure > /dev/null 2>&1 && make -j$(nproc) > /dev/null 2>&1 && make install > /dev/null 2>&1 
cd ../libgcrypt-1.9.4
echo "Building and Installing libgcrypt..."
./configure > /dev/null 2>&1 && make -j$(nproc) > /dev/null 2>&1 && make install > /dev/null 2>&1 
cd ..
echo "Building gpg-fingerprint-filter-gpu..."
make -j$(nproc)

SESSION_PREFIX="pgp_"
for ((i=1; i<=4; i++)); do
    SESSION_NAME="${SESSION_PREFIX}${i}"
    TASK="./gpg-fingerprint-filter-gpu -j 34 -a ed25519 -t 59654321 -m Y \"x{10}|x{11}|x{12}|x{13}|x{14}|x{15}|x{16}|x{4}y{8}\" /root/fs/output.pgp"
    screen -dmS "$SESSION_NAME"
    screen -S "$SESSION_NAME" -X stuff "$TASK\n"
    echo "Started task in screen session $SESSION_NAME"
done
sleep 3
TASK="python $(pwd)/ca.py /root/fs/output.pgp/ /root/fs/dist/ >> /root/fs/log.txt"
echo "#!/bin/bash" >> ./cron.sh
echo "while true; do" >> ./cron.sh
echo "$TASK" >> ./cron.sh
echo "sleep 1" >> ./cron.sh
echo "done" >> ./cron.sh
chmod +x ./cron.sh
bash ./cron.sh &