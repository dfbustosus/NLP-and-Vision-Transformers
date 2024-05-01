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

This sublayer converts the input tokens to vectors of dimension $d_{model}= 512$ or to the specified size of the model from past learning.

![Embedding](Input_embedding.jpg "Input Embedding")

The `tokenizer` will transform the sentence into `tokens`. Each `tokenizer` has own methods such as `Byte Pair Encoding (BPE)`, word piece and sentence piece methods. The original Transformer uses `BPE` but other models use different mehtods

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

So therefore, we can conclude that the input to the Transformers arquitecture is not just simple numbers. Models have learned word embeddings that already provide information regarding words association. However there is no context about the position of each word in the sequence. In order to solve this the next structure is vital the `positional encoding`

# 2. Encoder 
## 2.1 Positional Embedding

When we go out from the input embedding we have a sequence of vectors of dimension $d_{model}=512$. However we need to know the position of each word in the sequence and that's the `Input Embedding` purpose. The main idea is add a positional encoding value to the input embedding instead of having additional vectors to describe the position of a token in a sequence.

![Positional Embedding](Positional_embedding.jpg "Positional Embedding")

Basically the idea is add a value to the word embedding of each word so that is has that information. In that case we need to add a value to the $d_{model}=512$ dimensions. For each word embedding vector we need to find a way to provide information $j$ in the `range(0,512)` dimensions of the word embedding vector of each word.


## 2.2 Positional Encoding
You need to take in count that there are different ways to do this, one technique is the unit sphere with sine and cosine values. Vawani et al. (2017) provide the following functions to generate different frequencies for the **positional encoding(PE)** for each position and each dimension $j$ of the $d_{model}=512$ of the word embedding vector

$$PE_{pos_{2j}}= sin(\frac{pos}{10000^{\frac{2j}{d_{model}}}})$$

$$PE_{pos_{2j+1}}= cos(\frac{pos}{10000^{\frac{2j}{d_{model}}}})$$

The sine function will be applied to the even numbers and the cosine to the odd numbers. Some other cases do it differently. In our case the domain of the sine function is $j \in [0,255]$ and for the cosine function is $j \in [256,512]$

The function provided by Vaswani et al. (2017) is given by:

```python
def positional_encoding(pos,pe):
  for i in range(0, 512,2):
    pe[0][i] = math.sin(pos / (10000 ** ((2 * i)/d_model)
    pe[0][i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model
return pe
```
You can check how it looks the plot in `positional_encoding_simple.py` or `positional_encoding_simple.js`

Lets take an example for the phrase: 
*The black cat sat on the couch and the brown dog slept on the rug*

If we implement the  positional encoding to the `pos= 2` and `pos=10` for the words **black** and **brown** respectively we got the following results (check `positional_encoding_simple_2.py` or `positional_encoding_simple_2.js`):

```python
PE(2)=[[ 9.09297427e-01 -4.16146837e-01  9.58144376e-01 -2.86285442e-01
   9.87046251e-01 -1.60435961e-01  9.99164200e-01 -4.08766567e-02
   9.97479998e-01  7.09482514e-02  9.84702998e-01  1.74241229e-01
   9.63226623e-01  2.68690292e-01  9.35118300 ..............
   2.66704286e-08  1.00000000e+00  2.48187552e-08  1.00000000e+00
   2.30956397e-08  1.00000000e+00  2.14921566e-08  1.00000000e+00]]
```

And 
```python
PE(10)=[[-5.44021111e-01 -8.39071529e-01  1.18776483e-01 -9.92921018e-01
   6.92634182e-01 -7.21289047e-01  9.79174779e-01 -2.03019092e-01
   9.37632744e-01  3.47627440e-01  6.40478017e-01  7.67976503e-01
   2.09077004e-01  9.77899180e-01 -2.37917679e-01 ..............
   2.66704286e-08  1.00000000e+00  2.48187552e-08  1.00000000e+00
   2.30956397e-08  1.00000000e+00  2.14921566e-08  1.00000000e+00]]
```

You will see that if you calculate the cosine similarity between `pos(2)` and `pos(10)` you will get (check `positional_encoding_similarity_2.py` or `positional_encoding_similarity_2.js`): 
```python
cosine_similarity(pos(2), pos(10))= [[0.8600013]]
```

But if you calculate the cosine similarity using the OpenAI model **text-embedding-3-small** you got a different result (check `positional_encoding_similarity_1.py` or `positional_encoding_similarity_1.js`): 

```python
cosine_similarity(black, brown)= [[0.56902884]]
```
So therefore with these examples you will notice that the positional encoding is crucial for incorporating the notion of word order and distance into the model's understanding of the input sequence (an unique representation for each position in the input sequence). It ensures that the model can differentiate between words based not only on their content but also on their position in the sequence.

Now the big question is **How we add positional encoding to the embedding vector?**. Let's see it.

**Adding the positional encoding to the embedding vector**

Well basically adding the positional encoding vector to the word embedding vector as you can see in the following figure

![Adding positional encoding](Adding_positional_encoding.jpg "Adding positional encoding")

So the notation would be:

$$pc(word) = y_{word} + pe(position)$$

In the case of our last example for the word `black` we have:
$pc(black) = y_{black} + pe(2)$

However is we apply the solution like this we might lose information of the word embedding since could be minimized by the positional encoding vector. In order to solve this there are different ways to increase the value of $y$ to ensure that the information of the word embedding can be used properly, for example we can add an arbitrary value to $y$ like this:

$$y*\sqrt{d_{model}}$$

Thus, the general code structure would be the following:

```python
for i in range(0, 512,2):
  pe[0][i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
  pc[0][i] = (y[0][i]*math.sqrt(d_model))+ pe[0][i]
  pe[0][i+1] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
  pc[0][i+1] = (y[0][i+1]*math.sqrt(d_model))+ pe[0][i+1]
```

You can test it out using `adding_positional_encoding.py` or `adding_positional_encoding.js` The result should be

```python
[[1.0933194359378953, 0.26627445503091957, 0.3648030077827548, 0.28991768982320787, 0.8957610107050262,....,
  1.7605277430867114, -0.2971715378171267, 1.848602349594715, 0.14818397844827294, 1.1748635128654696]]
```

And if we find the cosine similarity between `pc(black)` and `pc(brown)` we got this (cehck out `adding_positional_encoding_2.py` or `adding_positional_encoding_2.js`)

*Cosine similarity between pc(black) and pc(brown): 0.8123017449234047*

So in summary we calculate this:

- `word similarity` 
```python
cosine_similarity(black, brown)= [[0.56902884]]
```
- `positional encoding vector similarity` between positions 2 and 10 
```python
cosine_similarity(pos(2), pos(10))= [[0.8600013]]
```
- `final positional encoding similarity` 
```python
cosine_similarity(pc(black), pc(brown))= [[0.81230]]
```

From the `positional encoding` each word contains the initial word embedding info and the positional encoding values. The output from the `positional encoding` is passed now to the `Multi-head attention sub-layer` let's go inside of this new `sub-layer`

## 2.3 Multi-Head Attention

![Multi head attention](Multi_head_attention.jpg "Multi head attention")

This sublayer has eigh heads and is followed by a post-layer normalization. We add residual connections to the output of the sublayer and normalize it. So we need to understand first how the `attention layer` works and therefore we can create a `multi-attention` structure and finally we can introduce the `post-layer normalization` concept

So let's start from the begin. The input to this sublayer is a vector which contains the embedding and the positional encoding of each word. The output of each sublayer feed the other layers. 

So the dimension of the vector of each word $x_n$ of an input sequence if $d_{model} =512$ for example is:

$$PE(X_n)= [d_{1},d_{2},....,d_{512}]$$

Each word become a vector of $d_{model}=512$

Also each word needs to be mapped to all the other words in order to determine how it fits in a sequence. For example let's say that we have this sentence: *"The dog is fine above the bed and it was cleaned"*

So for instance the model needs to be trained to determine if `it` is related to `dog` or `bed`. We can do this with a lot of computation using the $d_{model}=512$ but this is not feasible when we have a lot of sequences.

A smarter approach would be for the 8 heads of the model project the $d_{model}=512$ dimensions of each word $x_n$ in sequence $x$ into the $d_k = 64$ dimensions. We can run the eight heads in parallel to speed up the training and obtain 8 different representations subspaces of how each word relates to another

![Multi head representation](Multi_head_representation.jpg "Multi head representation")

So there are $8$ heads running in parallel. For example one head could decide that `it` is strongly related to `dog`, other head that `bed` is related to `cleaned` and so on. The output of each head is a matrix $Z_i$ with a shape $x* d_k$. And the output is defined as:

$$Z= (Z_0, Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7)$$

$Z$ must be concatenated to ensure that the output of the multi-head sublayer is not a sequence of dimensions. Before the output of the multi-head attention sublayer, the elements of Z are concatenated:

$$Multihead(out) = Concat(Z_0,Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7)= x, d_{model}$$

Each head is concatenated into $z$ which has a dimension $d_{model}=512$. 

Inside of each `head` $h_n$ of the attention mechanism the "word" matrices have three representations

1. A query matrix ($Q$) with dimension $d_q = 64$ which seeks all the key-value pairs of the other "word" matrices
2. A key matrix ($K$) with a dimension $d_k =64$
3. A value matrix ($V$) with a dimension $d_v =64$

The `Attention` is defined as the scaled dot-product attention, which is represented by the following equation

$$Attention (Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

The matrices have the same dimensions, so apply the scaled dot product is not complex, and concatenate the results is easy too. 

To obtain the $Q$, $K$ and $V$ matrices we need to train the model with corresponding $Q_w$, $K_w$ and $V_w$ with $d_k =64$ columns and $d_{model}=512$ rows. $Q$ is obtained by a dot product between $x$ and $Q_w$ and $Q$ with have a dimension of $d_k =64$.

You need to notice that you can modify these parameters as you like for example you can modify the `number of layers`, `heads`, $d_{model}$, $d_k$ and other ones to fit your model. But you need to be careful if you are going to change it. And also **don't worry** you don't need to do this all by hand but in order to understand how Transformers work we need to do one example at least one time to learn the concepts. So let's do an example to check how the `Attention mechanism` works. 