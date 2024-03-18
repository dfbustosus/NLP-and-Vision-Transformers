# Transformers

## What are?

**LLM = Large Language Models (LLMs)** are designed for parallel computing solving a wide range of tasks without fine tuning 

These models can perform self-supervised learning on big amount of data using complex arquitectures with billion or trillion of parameters

## Services including LLM
- Google Cloud AI
- AWS
- Azure
- OpenAI
- Google Colab
- Github Copilot
- HuggingFace
- Meta
- A lot more!

# O(1) time complexity

O(1) means order of 1 and this represents a constant time complexity. Order of 1 means order of 1 operation

To understand this lets take an example. If we use the following sentence

"David likes apples in the afternoon bot not in the morning"

This sentence has $n=11$. 

Any problem of language understanding depend on one word context. Thus implies that a word usually cannot be defined withouth an appropiate context. The meaning of a word can change in different contexts even that there is only one definition in the dictionary!!

# Attention layer

If we see careful the word "apples" has several semantic relationships 

- Dim 1: Association between apples and afternoon
- Dim 2: Association between apples and morning
- Dim 3: Association between David and apples
- Dim 4: Association between David and afternoon
- Dim 5: Association between David and morning
- ...........
- Dim Z

In the before case you see that `relationships` are defines pairwise (one word to another). That's exactly how `self-attention` works in any transformer model like `GPT`

The `O(1)` problem is just the following> We perform **one** operation per word `O(1)` for each word to find the relationship with another word in a pairwaise analysis

If we create a more formal notation then we have>

- `n=` represent the length og the sequence, in this case 11
- `d=` represent the number of dimensions expressed in floats. For example if `y` is a word we can represent it in a vector like this `[0.2333, -0.4300, 0.9566, ..., -0.2455]`. Now probably you are thinking why this vector and not other. Well the answer is because the model learns that the word `y` is represented by this after analyze a lot of text data points

When we talk about `O(1)` we refer to the `memory complexity`

Therefore the total computational complexity of an attention layer is given by
$O(n^2 * d)$. Where
- $n^2$ is the pairwise (word-to-word) operation of the whole sequence n
- $d$ represent the number of dimensions learned by the model e.g 512

# Recurrent layer

Contrary to the `Attention Layer`, Recurrent layers not operate similar. They are $O(n)$ and thats because **the longer the sequence, more memory will be consumed**. Why? Because they do not learn the diemensions with pairwise relationships, they learn in a sequence e.g

- Dim a: David
- Dim b: likes and David
- Dim c: apples and likes and David
- Dim d: in and apples and likes and David
- .............
- Dim z

As you can see for each word we dont just look another word, instead we lookk several other words at the same time. Therefore the number of dimensions for one word$d$ is multiplied by the dimensiones of a preceding word learning $d^2$ dimensions and the total computational complexity will be $O(n*d^2)$

# Computational time complexity (Attention Layer)

- The Attention layer use $O(1)$ memory time complexity enabling a total time complexity of $O(n^2*d)$ to perform a dot product between eaxh word. That implies multiply the representation $d$ of each word by another word. **Attention layers allow learn all the relationships in one matrix multiplication**

- Recurrent layers present a total time complexity of $O(n*d^2)$ given the $O(n)$ individual complexity. then perform same task as attention layer will require more operations.

We can create simulations regarding attention complexity and recurrent time complexity using a conceptual approach and test the results with CPU, GPU and TPU

1. **CPU = Central Processing Unit** - Primary component of computer. Is not efficent for large computation calculations
2. **GPU = Graphics Processing Unit** - Specialized unit for 3D image rendering and complex ML tasks such as matrix multiplications
3. **TPU = Tensor Processing Unit** - accelerating ML workload processor created by google and optimized by Tensorflow

# Results from computational simulations
1. Attention layer computational time complexity is much faster than hte recurrent layer computation titme complexity
2. Attention laey one-to-one word analysis makes suitable at detecting long-term dependencies
3. Attention layers enables matrix multiplication with a great advantage of GPUs and TPUs

The complexity $O(1)$ which derives to the $O(N^2 *d)$ complexity is the fundamental principle behind any LLM no matter which is.