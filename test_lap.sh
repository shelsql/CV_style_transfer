# CUDA_VISIBLE_DEVICES=0 
python ./test_adain.py\
  --content_dir ./input/content\
  --style_dir ./input/style\
  --decoder ./models/decoder_16.pth\
  --output ./results/lap\
  --alpha 1

  