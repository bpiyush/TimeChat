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


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/timechat.yaml', help="path to configuration file.")
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


def get_video_path(video_dir, video_id, ext="webm"):
    from glob import glob
    paths = glob(os.path.join(video_dir, f"*/{video_id}.{ext}"))
    assert len(paths) == 1
    return paths[0]


def get_llm_answer(
        video_path=None, text_options=None, num_beams=1, temperature=1.0, debug=False,
    ):
    if video_path is None:
        video_path = "../TimeBound.v1/sample_data/folding_paper.mp4"
    if text_options is None:
        text_options = [
            "Someone folding a paper.",
            "Someone unfolding a paper.",
        ]
    assert os.path.exists(video_path)
    assert len(text_options) == 2


    # Check normal prompt
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that "\
        "the user provides."\
        "Follow the instructions carefully and explain your answers in detail."
    img_list = []
    llm_message = chat.upload_video_without_audio(
        video_path, chat_state, img_list, video_loader="load_video_cv2", n_frms=96,
    )


    user_message = f"""
    Given this video, you have to select which is the option that correctly describes the video. (a) {text_options[0]} (b) {text_options[1]}. You have to only answer (a) or (b).
    """
    if debug:
        print(user_message)
    chat.ask(user_message, chat_state)


    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    if debug:
        print(llm_message)

    correct_answer = f"(a) {text_options[0]}"
    is_correct = int(correct_answer in llm_message)


    # Check reversed prompt
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that "\
        "the user provides."\
        "Follow the instructions carefully and explain your answers in detail."
    img_list = []
    llm_message = chat.upload_video_without_audio(
        video_path, chat_state, img_list, video_loader="load_video_cv2", n_frms=96,
    )


    user_message = f"""
    Given this video, you have to select which is the option that correctly describes the video. (a) {text_options[1]} (b) {text_options[0]}. You have to only answer (a) or (b).
    """
    chat.ask(user_message, chat_state)


    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]

    correct_answer = f"(b) {text_options[0]}"
    is_correct += int(correct_answer in llm_message)

    accuracy = is_correct / 2.
    return accuracy


if __name__ == "__main__":

    print("[:::] Loading model.")
    model, vis_processor, args, chat = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params/1e9}B")


    # Debug
    debug = False
    if debug:
        is_correct = get_llm_answer()
        print(f"Accuracy: {is_correct:.2f}")

    # Load data
    print("[:::] Load data.")

    csv_path = "/scratch/shared/nfs2/piyush/datasets/SSv2/metadata/time_antonyms-validation.csv"
    df = pd.read_csv(csv_path)

    data_dir = "/scratch/shared/beegfs/shared-datasets/SomethingSomething-V2/"
    video_dir = os.path.join(data_dir, "videos")

    iterator = tqdm(df.iterrows(), total=len(df))
    text_corrects = []
    failed = []
    for i, row in iterator:
        row = row.to_dict()
        video_path_x = get_video_path(video_dir, row["id_x"])
        video_path_y = get_video_path(video_dir, row["id_y"])
        label_x = row["label_x"]
        label_y = row["label_y"]

        try:
            # Check for first video
            is_correct = get_llm_answer(video_path_x, [label_x, label_y], debug=debug)
            text_corrects.append(is_correct)

            # Check for second video
            is_correct = get_llm_answer(video_path_y, [label_y, label_x], debug=debug)
            text_corrects.append(is_correct)
        except:
            failed.append(i)

        if debug:
            if i == 10:
                break
    
    print("Number of failed: ", len(failed))

    text_corrects = np.array(text_corrects)
    print("Video to text accuracy: ", text_corrects.mean())