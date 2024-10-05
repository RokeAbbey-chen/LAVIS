
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

from lavis.processors.blip_processors import Blip2ImageTrainProcessor
from train import parse_args, get_runner_class
from PIL import Image
import numpy as np

def main0():
    job_id = now()
    cfg = Config(parse_args())
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg).cuda()
    image = Image.open("/dataset/coco2014/train2014/COCO_train2014_000000157119.jpg")
    image = image.convert('RGB')
    # image = torch.from_numpy(np.array(image))
    vis_processor = Blip2ImageTrainProcessor(224)
    image = vis_processor(image).cuda()
    image = torch.stack([image, image])
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.generate({'image': image})
    print(outputs)
    

    # runner = get_runner_class(cfg)(
    #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    # )
    # runner.train()
    
    
main0()