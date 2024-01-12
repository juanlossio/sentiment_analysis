# Sentiment Analysis

## A) Baseline Applications:

- All baseline applications such as LIWC2015, SentiStrength, TextBlob, VADER, Stanza, TweetEval, Pysentimiento, and NLPTown, can be accessed through their respective websites. These represent distinct software packages, each with its own installation requirements.

   It is important to note that LIWC2015 is the only tool associated with licensing fees.

## B) OPT: Open Pre-trained Transformer Language Models

- We are sharing the code for fine-tuning the OPT models, along with their respective hyperparameters in `opt-train.py`.

   However, this process may require the use of multiple GPUs for preparation and deployment.

## C) ChatGPT

- It is important to mention that utilizing ChatGPT through its API for automated use requires an access key and a credit card.
- Additionally, the APIs have already undergone substantial changes since the time of our paper's writing.
- Therefore, we have decided to provide instructions on how to use the most recent ChatGPT API to automatically generate code for predicting the sentiment of every sentence. The process is as follows:

   1. Visit [OpenAI Playground](https://platform.openai.com/playground);
   2. Select "Chat" mode;
   3. Use the following prompt: "What is the sentiment of the following sentence 'x'";
   4. Once one interaction has been completed, the "View Code" button will provide API code to reproduce it, which can be adapted for bulk processing.


## Citation
```
@article{weger2023trends,
  title={A Comparison of ChatGPT and Fine-Tuned Open Pre-Trained Transformers (OPT) Against Widely Used Sentiment Analysis Tools: Sentiment Analysis of COVID-19 Survey Data},
  author={Lossio-Ventura JA, Weger R, Lee AY, Guinee EP, Chung J, Atlas L, Linos E, Pereira F},
  journal={JMIR Mental Health},
  volume={e},
  number={1},
  pages={e40899},
  year={2024},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
```
