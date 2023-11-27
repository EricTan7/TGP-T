# TGP-T🚀: Compound Text-Guided Prompt Tuning via Image-Adaptive Cues

<p align="center">
    <img src="src/method.png" alt="TGP-T Framework">
</p>



Vision-Language Models (VLMs) such as CLIP have demon- strated remarkable generalization capabilities to downstream tasks. However, existing prompt tuning based frameworks need to parallelize learnable textual inputs for all categories, suffering from massive GPU memory consumption when there is a large number of categories in the target dataset. Moreover, previous works require to include category names within prompts, exhibiting subpar performance when dealing with ambiguous category names. To address these shortcom- ings, we propose Compound Text-Guided Prompt Tuning (TGP-T) that significantly reduces resource demand while achieving superior performance. We introduce text supervi- sion to the optimization of prompts, which enables two ben- efits: 1) releasing the model reliance on the pre-defined cat- egory names during inference, thereby enabling more flexi- ble prompt generation; 2) reducing the number of inputs to the text encoder, which decreases GPU memory consump- tion significantly. Specifically, we found that compound text supervisions, i.e., category-wise and content-wise, is highly effective, since they provide inter-class separability and cap- ture intra-class variations, respectively. Moreover, we con- dition the prompt generation on visual features through a module called Bonder, which facilitates the alignment be- tween prompts and visual features. Extensive experiments on few-shot recognition and domain generalization demon- strate that TGP-T achieves superior performance with con- sistently lower training costs. It reduces GPU memory usage by 93% and attains a 2.5% performance increase on 16-shot ImageNet.



**Code will be released soon.**
