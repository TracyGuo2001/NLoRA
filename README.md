# NLoRA
NLoRA: Nyström-Initiated Low-Rank Adaptation for Large Language Models

## Introduction
Parameter-efficient fine-tuning (PEFT) is essential for adapting large language models (LLMs), with low-rank adaptation (LoRA) being the most popular approach. However, LoRA suffers from slow convergence, and some recent LoRA variants, such as PiSSA, primarily rely on Singular Value Decomposition (SVD) for initialization, leading to expensive computation. To mitigate these problems, we use the Nyström method, which follows a three-matrix manipulation.

We first introduce Structured LoRA (SLoRA), which investigates adding a small intermediate matrix between the low-rank matrices A and B. Secondly, we propose Nyström LoRA (NLoRA), which leverages Nyström-based initialization for SLoRA to improve its effectiveness and efficiency. Finally, we propose Intermediate Tune (IntTune), which explores fine-tuning exclusively on the intermediate matrix of NLoRA to further boost LLM efficiency.

We evaluate our methods on five natural language generation (NLG) tasks and eight natural language understanding (NLU) tasks. On GSM8K, SLoRA and NLoRA achieve accuracies of 56.48% and 57.70%, surpassing LoRA by 33.52% and 36.41%, with only 3.67 million additional trainable parameters. IntTune improves average NLG performance over LoRA by 7.45% while using only 1.25% of its parameters. These results demonstrate the efficiency and effectiveness of our approach in enhancing model performance with minimal parameter overhead.


![structure](./assets/structure.jpg)
![nystrom](./assets/nystromInitialization.png)
![NLG](./assets/nlg.jpg)
![intTune](./assets/intTune.jpg)
