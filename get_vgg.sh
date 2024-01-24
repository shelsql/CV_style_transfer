mkdir models
cd models

URL="https://docs.google.com/uc?export=download&id=1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU" -O vgg_normalized.pth && rm -rf /tmp/cookies.txt
