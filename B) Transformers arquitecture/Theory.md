# Transformers

The original structure has **6 layers** and each layer contains **sublayers**. So in general there are 6 layers in the **encoder** stack and other 6 in the **decoder** stack. But don't worry we will mention what means these concepts

<img src="Transformer_arquitecture.jpeg" alt="Transformers" title="Transformers Arquitecture" width="600" height="800">

As you can see above there is no RNN, LSTM or CNN arquitecture and thats because recurrence is no longer important in this type of arquitecture

The `attention mechanism` is a `word-to-word` operation (in reality is `token-to-token` but let's keep it simple). And this `attention mechanism` determines how each word is related to all other words in a sequence including the word analyzed.

**Example**
Let's use the following sentence

*The cat sat on the mat*

When we refer to `attention` means that we will run dot products between word vectors and therefore determine what are the strongest relationships between a given word and all the other words including itself (e.g sat and sat)

![Similarity](Similarity_words.jpg "Associations between words")

Therefore the `attention mechanism` provide a deeper relationship between words. For each attention sublayer the original model runs 8 attention mechanisms in parallel to speed up calculations, thats called `multi-head attention`.

Some benefits of this are:
1. A complete analysis of sequences
2. Avoid recurrence which reduce calculation operations
3. Parallelization reducing training time
4. Each attention mechanism learns different perspectives of the same input sequence

# Encoder Stack (general description)
The encoder stack has $N=6$ layers and each layer hasta the following structure:

![Encoder](Encoder_stack.jpg "Encoder Stack")

Each layer contains two main sublayers
- A multi-headed attention mechanism
- Fully connected position-wise feedforward network

Each sublayer is surround by a residual connection. These connections transport unprocessed input $x$ of a sublayer to a layer normalization function. Therefore the normalized output of each layer is:

$$LayerNormalization(x+Sublayer(x))$$

The structure of the N=6 layers is identical but the content is different since there are different weights. The only exception is the first layer which includes the `embedding layer` at the bottom, the others 5 layers not conatin this (beacuse the encoded input can't change). 

Take in count that although all the multi-head attention mechanisms perform the same functions they do not perform the same tasks. The reason is because each leayer learns from previous layers checking different ways of associating the tokens in the sequence

To avoid problems of dimensionality we can set a dimension size to represent tokens, a constant for a given model for example in the original Transformer model $d_{model} = 512$

# 1. Encoder - Input Embedding

this sublayer converts the input tokens to vectors of dimension $d_{model}= 512$ or to the specified size of the model from past learning.

![Embedding](Input_embedding.jpg "Input Embedding")

The `tokenizer` will transform the sentence into `tokens`. Each `tokenizer` has own methods such as `Byte Pair Encoding (BPE)`, word piece and sentence piece methods. The original Transformer uses BPE but other models use different mehtods

You can test it out in OpenAI [Tokenizer openAI](https://platform.openai.com/tokenizer) you can put the following sentence *the Transformer is an innovative NLP model!* and you will get this:

![Tokenizer](Tokenizer.jpg "Tokenizer")

As you can see there are 9 tokens and 43 characters. You must know that usually each word is represented by an integer in the tokenizer.

Additionally each word is represented by a $d_{model}=512$ dimensions, for example:

```bash
curl https://api.openai.com/v1/embeddings   -H "Content-Type: application/json"   -H "Authorization: Bearer sk-XXXXXXXX"   -d '{
    "input": "The black cat sat on the couch and the brown dog slept on the couch",
    "model": "text-embedding-3-small"
  }'

```

You can also test it with the files `input_embedding.py` or `input_embedding_node.js` if you prefer Python or NodeJS. The output will be something like this for the phrase (Note: ensure that you have an `OPENAI_API_KEY` in a `.env` file to avoid issues):

```python 
array=[-0.027019580826163292, -0.03926384449005127, -0.03197353333234787, 0.007338985800743103, 0.037857700139284134, ......, -0.03318497911095619, 0.01930742710828781]
```

You can also obtain a similar representation for each word. And we can also check if two words are similar or not. For example que can check if the words `king` and `queen` are similar using the Cosine distance metric. You can run the `cosine_similarity.py` or `cosine_similarity_node.js` as you prefer to get the ouput.

*Cosine similarity between 'king' and 'queen': 0.5905304590968364*

So therefore we can conclude that the input to the Transformers arquitecture is not just simple numbers. Models have learned word word embeddings that already provide information regarding words association. However there is no context about the position of each word in the sequence. In order to solve this the next structure is vital the `positional encoding`