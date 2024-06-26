# FAQ

## 1. The performance on YouCook2 is low

Please check that if you correctly transform the video to `"youcook2_6fps_224"` (see https://github.com/RenShuhuai-Andy/TimeChat/blob/master/docs/DATA.md#compressing-videos).

## 2. The performance of the released ckpt is lower than the numbers on the paper

Our released ckpt is different from the version used in the paper. The released ckpt was trained after cleaning the code and fixing a minor bug in QuerYD instructions data (some videos have the same start and end timestamps in the raw annotations file, so we only use one timestamp in the revision).

In our evaluation, the performance of the released ckpt on YouCook2 is higher than that in the paper, while the performance on Charades-STS & QVHighlight is lower. We also note that the output generated by LLM is different each time, which may cause fluctuations in the evaluation results.

We have uploaded the ckpt used in our paper, please refer to https://huggingface.co/ShuhuaiRen/TimeChat-7b-paper. With this ckpt, We believe you can reproduce the results in our paper.

## 3. How to better instruct the model to perform QA or other specialized tasks?

Due to the use of a large amount of temporal localization task-related data during instruction-tuning, TimeChat-7B may lose some language capabilities. 
Specifically, even when you ask questions unrelated to temporal localization, TimeChat may occasionally generate responses like `"this happens at xx-xx seconds."` 
This can reduce TimeChat's performance on some benchmarks (which are often designed as multiple-choice questions). 

To mitigate this issue, we can reduce [`lora_alpha`](https://github.com/RenShuhuai-Andy/TimeChat/blob/master/timechat/models/timechat.py#L180) during inference to control the mixing ratio of LoRA weights and the original LLM weights. 
The original `lora_alpha` is set to **32**, meaning the LoRA weights are fully applied to the LLMs. 
By lowering it to **20**, the proportion of LoRA weights introduced is reduced, which can be very helpful to improve the quality of generated responses (ack: Post Optimization in [PLLaVA](https://arxiv.org/abs/2404.16994)).

```python
if self.lora:
    logging.info('Using LORA')
    from peft import LoraConfig, get_peft_model, TaskType
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=lora_inference_mode,
        r=32,
        lora_alpha=20,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    self.llama_model = get_peft_model(self.llama_model, config)
    self.llama_model.print_trainable_parameters()
```

## 4. Implementation of the time-aware frame encoder

The current implementation of the time-aware frame encoder has two differences from InstructBLIP:

(1) The tokenizer for timestamps is different (LLamaTokenizer vs. BertTokenizer) (https://github.com/RenShuhuai-Andy/TimeChat/issues/19).

(2) The output sequence contains timestamp tokens (https://github.com/RenShuhuai-Andy/TimeChat/issues/28) 
Our current output sequence contains both visual and timestamp tokens. According to InstructBLIP, the timestamp tokens should be removed after the encoder.

We are not sure if these two differences will have a negative impact on the performance, but we will take a look at it if we have time.


## 5. Does TimeChat support Chinese?

Unfortunately, our model currently only supports English. It seems that it can understand Chinese questions, but cannot generate responses in Chinese. 
See https://github.com/RenShuhuai-Andy/TimeChat/issues/25#issuecomment-2102594079.