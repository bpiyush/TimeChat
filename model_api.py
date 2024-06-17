"""Simplified API access to the model."""
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
import pandas as pd
from tqdm import tqdm
from termcolor import colored

from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import (
    Chat, Conversation, default_conversation,SeparatorStyle, conv_llava_llama_2
)

import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import (
    ToTHWC, ToUint8, load_video, load_video_cv2
)
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr


def setup_seeds(seed):
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    curr_dirpath = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(curr_dirpath, "eval_configs/timechat.yaml")
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default=cfg_path, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--text-query", default="What is he doing?", help="question the video")
    parser.add_argument("--video-path", default='examples/hotdog.mp4', help="path to video file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args(args=[])
    return args


def load_model():
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    ckpt_root = "/work/piyush/pretrained_checkpoints/LargeModels/TimeChat/"
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = f"{ckpt_root}/TimeChat-7b/timechat_7b.pth"

    model_config.llama_model = os.path.join(ckpt_root, "Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf/")

    model_config.vit_model = os.path.join(ckpt_root, "eva_vit_g.pth")
    model_config.q_former_model = os.path.join(ckpt_root, "instruct_blip_vicuna7b_trimmed.pth")

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    return model, vis_processor, args, chat


def ask_about_video(chat, video_path, user_message, num_beams=1, temperature=1.0):
    # Check normal prompt
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that "\
        "the user provides."\
        "Follow the instructions carefully and explain your answers in detail."
    img_list = []
    llm_message = chat.upload_video_without_audio(
        video_path, chat_state, img_list, video_loader="load_video_cv2",
    )


    # user_message = f"""
    # Given this video, you have to select which is the option
    # that correctly describes the video.
    # (a) {text_options[0]} (b) {text_options[1]}

    # You have to only answer (a) or (b).
    # """
    chat.ask(user_message, chat_state)


    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    return llm_message
