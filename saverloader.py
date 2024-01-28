import torch
import os, pathlib
import numpy as np

def load(ckpt_dir, model, model_name='decoder'):
    print('reading ckpt from %s' % ckpt_dir)
    if not os.path.exists(ckpt_dir):
        print('...there is no full checkpoint here!')
        print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_dir --')
    else:
        ckpt_names = os.listdir(ckpt_dir)
        steps = [int((i.split('_')[2]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = '%s_iter_%d.pth.tar' % (model_name, step)
            path = os.path.join(ckpt_dir, model_name)
            print('...found checkpoint %s'%(path))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint, strict=False)
                
        else:
            print('...there is no full checkpoint here!')
    return step
