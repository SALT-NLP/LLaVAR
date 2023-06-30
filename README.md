# LLaVAR

**LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding**

*Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, Tong Sun*

[Project Page](https://llavar.github.io/)

[Arxiv Link](https://arxiv.org/abs/2306.17107)

![alt text](./images/teaser.png "LLaVAR")

```
@misc{zhang2023llavar,
      title={LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding}, 
      author={Yanzhe Zhang and Ruiyi Zhang and Jiuxiang Gu and Yufan Zhou and Nedim Lipka and Diyi Yang and Tong Sun},
      year={2023},
      eprint={2306.17107},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

[UPDATE 06/29] Initial Release.

*The main difference between our code and LLaVA's code is that we modified the training/testing/serving files to support Vicuna v1.1, which use '\</s\>' as the seperator instead of '###'.*

## Environment Setup

**Please preprare the environment/merge the model weight following LLaVA.**

Model Weight Delta: [Google Drive](https://drive.google.com/drive/folders/19uEwM1VrzX_KqCzzSJAh8RqOHbf4WS5Z?usp=sharing)

## Training Data

Our image data is already transformed into the format of LLaVA pretraining/finetuning (They have "fake" file name in the format of CC3M and COCO), you can download them and merge them into the LLaVA training sets.

Our instructions, on the otherhand, already contains LLaVA's instructions.

Pretraining Images： [Google Drive](https://drive.google.com/file/d/1zWpqnAcaG_dUwkJJUvP9FH9zq__c-ODY/view?usp=sharing)

Pretraining Instructions (585K + 422K)： [Google Drive](https://drive.google.com/file/d/1_GCHFwrPGjp-9tZlDBwVkdz-L1ymchKY/view?usp=sharing)

Finetuning Images： [Google Drive](https://drive.google.com/file/d/1_GCHFwrPGjp-9tZlDBwVkdz-L1ymchKY/view?usp=sharing)

Finetuning Instructions (158K + 16K): [Google Drive](https://drive.google.com/file/d/1ISdKOV1wwVkLHf5FNutctpOBa-CmNRFv/view?usp=sharing)

Finetuning Instructions (158K + 20K): [Google Drive](https://drive.google.com/file/d/1NHO8lly6pUo-fdyOAyWeGiQJWRb9qggk/view?usp=sharing)


## Evaluation Data
Evaluation Images： [Google Drive](https://drive.google.com/file/d/1tQQ6CX0fCH2kMuI9imrcEkYRWoVKScWX/view?usp=sharing)

GPT-4 Evaluation Contexts (585K + 422K)： [File](./files/caps_laion_50_val.jsonl)

GPT-4 Evaluation Rules： [Google Drive](./files/rule_read_v3.json)

Questions: [Google Drive](./files/qa50_questions.jsonl)

GPT-4 Answers: [Google Drive](./files/qa50_gpt4_answer.jsonl)


## Training Script

You should merge our pretraining images into the cc3m folder.


```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
   /sensei-fs/users/yanzhez/LLaVA/llava/train/train_mem.py \
    --model_name_or_path /path/to/models/vicuna_13b_v1_1 \
    --data_path /path/to/chat_llavar.json \
    --image_folder /path/to/cc3m \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir /path/to/checkpoint \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --image_aspect_ratio 'pad' \
    --report_to wandb
```

You should merge our finetuning images into the coco2017 folder.


```Shell
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    /sensei-fs/users/yanzhez/LLaVA/llava/train/train_mem.py \
    --model_name_or_path /path/to/models/vicuna_13b_v1_1 \
    --data_path /path/to/llava_instruct_150k_llavar_16k.json \
    --image_folder /path/to/coco/images/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /path/to/mm_proj/llava-13b-pretrain.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /path/to/checkpoint \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 8000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --image_aspect_ratio 'pad' \
    --report_to wandb
```

Instruction-following on COCO images.

```
python /path/to/LLaVA/llava/eval/model_vqa.py \
    --model-name /path/to/checkpoint \
    --question-file \
    /path/to/LLaVA/playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014/val2014 \
    --answers-file \
    /path/to/qa90-answer-file.jsonl \
    --conv-mode "llava_v1"
```

Instruction-following on a given image URL.
```
python -m llava.eval.run_llava \
    --model-name /path/to/checkpoint \
    --image-file "https://cdn.shopify.com/s/files/1/0057/3728/3618/products/a-man-called-otto_ezrjr0pm_480x.progressive.jpg" \
    --query "Who starred in the movie?"
```


### Acknowledgement
The code base is mainly from the LLaVA project. You can also pay attention to the recent Vicuma model update.
