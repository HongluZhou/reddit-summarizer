# reddit-summarizer
Rutgers CS 2019 spring NLP course project
<!---
Google Folder: https://drive.google.com/drive/folders/1X9Z8pT9eW3bUdGQT7OO14Oqd5a2_kjgU?usp=sharing
--->
  
## Final Presentation: google sides [link](https://docs.google.com/presentation/d/1G_7menqhi7S85ShZDGjWppuFmY8TWyW9-CWyI5zE1ts/edit?usp=sharing)   
    
### Resources  
- [first presentation: Can we summarize Reddit post?](https://docs.google.com/presentation/d/1z4J6HMYFLQaS3g8zjvqCV3cazFWsUV9EiI55oUluVKE/edit?usp=sharing)   
- [Autotldr](https://www.reddit.com/r/autotldr/comments/31b9fm/faq_autotldr_bot/) is a bot that uses [SMMRY](https://smmry.com/about) to automatically summarize long reddit submission.   
- [text compactor tool](https://www.textcompactor.com/)    
- [TL;DR The abstractive summarization challenge](https://www.reddit.com/r/MachineLearning/comments/a6erpw/project_the_tldr_challenge/). Good dataset to use! An on-going challenge.   
- [What is the state of text summarization research?](https://www.reddit.com/r/LanguageTechnology/comments/94m0kw/what_is_the_state_of_text_summarization_research/).  
- [Datasets for text document summarization?](https://www.reddit.com/r/MachineLearning/comments/48wqey/datasets_for_text_document_summarization/)   
- [A Quick Introduction to Text Summarization in Machine Learning](https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f). Described the types of techniques.    
- [How to Clean Text for Machine Learning with Python](https://machinelearningmastery.com/clean-text-machine-learning-python/) .   
- [Attention in Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)   
- [A Brief Overview of Attention Mechanism](https://medium.com/syncedreview/a-brief-overview-of-attention-mechanism-13c578ba9129). It has good equations.    
- [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html). It has good equations.  
- [DeepInf: Social Influence Prediction with Deep Learning](A very good paper to understand attension mechanism).  
- [Graph Attension Networks](https://arxiv.org/pdf/1807.05560.pdf)   

### Tutorials & Tools
- [Encoder-Decoder Models for Text Summarization in Keras](https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/), [code](https://github.com/chen0040/keras-text-summarization).    
- [Text Summarization Using Keras Models](https://hackernoon.com/text-summarization-using-keras-models-366b002408d9)     
- [tensor2tensor](https://github.com/tensorflow/tensor2tensor)    
- [fairseq](https://github.com/pytorch/fairseq)    
- [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)   
- [Keras exmaple code: English to French](https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py)    
- [ml-notebooks](https://github.com/neonbjb/ml-notebooks/blob/master/keras-seq2seq-with-attention/keras_translate_notebook.ipynb)  
- [Keras BERT](https://pypi.org/project/keras-bert/)   
- [keras-seq2seq-with-attention](https://github.com/neonbjb/ml-notebooks/tree/master/keras-seq2seq-with-attention) Note - Tensorflow 1.13 and greater versions currently have problems with the code.  
        
### Metrics   
- [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)): TLDR challenge uses the F-1 scores accordingly for ROUGE-1, ROUGE-2 and ROUGE-LCS as quantitative evaluation.   
- Usually, a qualitative evaluation will be performed through crowdsourcing. Human annotators will rate each candidate summary according to five linguistic qualities as suggested by the [DUC guidelines](https://duc.nist.gov/pubs/2006papers/duc2006.pdf). - [Re-evaluating Automatic Metrics for Image Captioning](https://aclweb.org/anthology/E17-1019): This paper has good explanation for BLEU, METEOR, ROUGE, and CIDEr.     
   
### Papers   
- [A very useful collection of state-of-art related work (since 2015) and their implementations](https://tldr.webis.de/tldr-web/soa.html).
- [Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://github.com/ctr4si/MMN). Dataset provided but no implementation.      
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)  
- [Generating News Headlines with Recurrent Neural Networks](https://arxiv.org/abs/1512.01712)    
- [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). Proposed Seq2Seq.
- [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685).  
- [An Improved Phrase-based Approach to Annotating and Summarizing Student Course Responses](http://aclweb.org/anthology/C16-1006)  
   
### More thinking and future work notes
- Attention mechanism helps.   
- Comparison between character-level model and word-level model.    
- Could we use hidden vector of the model to serve as embedding vector of the text, and further do other tasks like subreddit classification etc?    
- LSTM methods require lots of training data. 
- Should compare to a baseline model that is not so statistically intensive, like latent dirichlet allocation, as well as using a generic classification method like BERT that's not tuned to the particular text that you are working with.  
