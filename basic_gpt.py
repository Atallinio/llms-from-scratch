# # Building LLMs from scratch

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import keras.ops as K
import re
import tiktoken # Byte Pair Encoding 
import matplotlib.pyplot as plt


# Load the shakespeare text dataset: contains 100,000 characters!

# In[2]:


dataset = tfds.load(name='tiny_shakespeare')

train = dataset['train']
for text in train:
    x = text['text'].numpy().decode('utf-8')
print(x[:100])
print(f"\nlength of the entire text file: {len(x)}")


# ## Setting up a custom simple tokenizer 
# Converts normal text into tokens using regex ----> then converts from tokens into token ids using a custom class

# In[3]:


tokens = re.split(r'([,.:;?_!"()\']|--|\s)', x)
print(tokens[:100])

vocabulary = sorted(set(tokens))

# Create a Dictionary with additional special tokens ("<|unk|>", "<|eos|>") 
# for an unkown word or the end of text (incase I train with multiple text sources).\

dictionary = {item:value for value, item in enumerate(vocabulary)}
dictionary["<|unk|>"] = len(dictionary)
dictionary["<|eos|>"] = len(dictionary)
#dictionary["<|bos|>"] = len(dictionary)
#dictionary["<|pad|>"] = len(dictionary)


# In[4]:


class SimpleTockenizer:
    """
    A simple tokenizer which using a dictionary converts the text into token ids 
    """
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.dictionary_reverse = {value:item for item, value in dictionary.items()}

    def encode(self, text):
        split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = list()
        for item in split:
            try: 
                tokens.append(self.dictionary[item])
            except:
                tokens.append(self.dictionary["<|unk|>"])
                
        return tokens

    def decode(self, tokens):
        text = "".join([self.dictionary_reverse[token] for token in tokens])
        return text


# In[5]:


tokenizer = SimpleTockenizer(dictionary)
tokens = tokenizer.encode(x)
print(tokens[:10])

text = tokenizer.decode(tokens)
print(text[:100])


# ## Using the GPT2 tokenizer form the tiktoken library
# The GPT2 tokenizer uses byte pair encoding which creates tokens for entire words and for sub-word characters

# In[6]:


tiktok = tiktoken.get_encoding("gpt2")
integers = tiktok.encode(x, allowed_special={"<|eos|>"})
print(integers[:50])

strings = tiktok.decode(integers[:50])
print(strings)


# ## Setting up the input and target values using the Windowing technique

# In[7]:


context_size = 4
for i in range(1, context_size+1):
    inputs = integers[:i]
    target = integers[i]
    print(tiktok.decode(inputs) + '------->' + tiktok.decode([target]))
    


# ## Create a Custom DataLoader to load the Data into TensorFlow

# In[8]:


class DataLoader:
    """
    A custom data loader which loads data using the tiktoken tokenizer
    into Tensorflow, creating input and target values using the windowing technique
    """
    def __init__(self, text, stride, max_length):
        self.input_ids = []
        self.target_ids = []

        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = tokenizer.encode(text, allowed_special={"<|eos|>"})
        
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(token_ids[i:i+max_length])
            self.target_ids.append(token_ids[i+1:i+1+max_length])
            
        # Convert lists to TensorFlow tensors
        self.input_ids = tf.convert_to_tensor(self.input_ids, dtype=tf.int32)
        self.target_ids = tf.convert_to_tensor(self.target_ids, dtype=tf.int32)
            
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    def load_data(self, batch_size=8, shuffle=False, buffer_size=1024):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_ids, self.target_ids)) 
        # Shuffle the dataset if required
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # Batch the dataset
        dataset = dataset.batch(batch_size)

        # Prefetch the dataset for better performance
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


# In[9]:


data = DataLoader(x, 2, 10)
dataset = data.load_data(batch_size=1)


# In[10]:


print(next(iter(dataset)))


# ## Creating a token embedding 

# In[32]:


max_length = 1024
stride = 8 
batch_size = 8

data = DataLoader(x, stride, max_length)
dataset = data.load_data(batch_size)
next(iter(dataset))[0]


# In[34]:


vocab_size = 50257          # Size of the token dictionary
output_dim = 768            # Size of the embedding
context_len = max_length    # To create the positional embedding

# Create an embedding for each token (A vector representation of that token)
token_embedding = keras.layers.Embedding(vocab_size, output_dim)(next(iter(dataset))[0])

# Create a positional embedding which has information of the word position
pos_idx = tf.range(context_len)
pos_embedding = keras.layers.Embedding(context_len, output_dim)(pos_idx)

# Add the positional information to the original token embedding
input_embedding = token_embedding + pos_embedding
input_embedding.shape

# ## Creating a Simple Self Attention Layer

# In[13]:


def Attention(inputs):
    """
    A very simple implementation of the Self Attention Layer
    """

    # 1. Calculate the relationship between each input and all other inputs in the sequence
    attention_scores = tf.matmul(inputs, tf.transpose(inputs))

    # 2. Normalize the attention scores for better learning (better for gradient descent)
    norm_as = keras.layers.Softmax()(attention_scores)

    # 3. generate the final context vector by multiplying each attention score with 
    # its corresponding input and suming them up
    final_context_vec = tf.matmul(norm_as, inputs)

    return final_context_vec


# In[14]:


input_embedding.shape


# In[ ]:


class SelfAttention(keras.Layer):
    def __init__(self, dim, bias=True):
        super().__init__()

        # The dimention of the query, key and value weights
        self.dim = dim
        self.bias = bias

    def build(self, input_shape):
        # Initializing the query, key and value weights
        """
        Use the keras weight matrices to initialize the weights 
            self.Wq = self.add_weight((input_shape[-1], self.dim), name="Wq")
            self.Wk = self.add_weight((input_shape[-1], self.dim), name="Wk")
            self.Wv = self.add_weight((input_shape[-1], self.dim), name="Wv")
        """        

        # Use the keras dense layer for the weights initialization
        self.Wq = keras.layers.Dense(self.dim, use_bias=self.bias)
        self.Wk = keras.layers.Dense(self.dim, use_bias=self.bias)
        self.Wv = keras.layers.Dense(self.dim, use_bias=self.bias)

    def call(self, inputs):
        #Calculate keys, queries and values vectors 
        """
        Calculate the vectors using dot product
            keys = K.dot(inputs, self.Wk)
            queries = K.dot(inputs, self.Wq)
            values = K.dot(inputs, self.Wv)
        """
        # Calculate the vectors using the dense layer which is functionaly the same as doing a dot product
        keys = self.Wk(inputs)
        queries = self.Wq(inputs)
        values = self.Wv(inputs)

        attention_score = K.dot(queries, K.transpose(keys))

        attention_weights = K.softmax(attention_score / self.dim**0.5, axis=-1)

        context_vector = K.dot(attention_weights, values)

        return context_vector


# In[81]:


class CasualSelfAttention(keras.Layer):
    def __init__(self, dim, context_length=100, bias=True, dropout=0.5):
        super().__init__()

        # The dimention of the query, key and value weights
        self.dim = dim
        self.bias = bias
        self.context_length = context_length
        self.dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape):
        # Initializing the query, key and value weights
        # Use the keras dense layer for the weights initialization
        self.Wq = keras.layers.Dense(self.dim, use_bias=self.bias)
        self.Wk = keras.layers.Dense(self.dim, use_bias=self.bias)
        self.Wv = keras.layers.Dense(self.dim, use_bias=self.bias)

        self.mask = self.mask()

    def mask(self):
        # Create & Apply a mask on the attention scores
        mask = K.triu(K.ones((self.context_length,self.context_length)), k=1) # Shape: (seq_len, seq_len)
        mask = K.expand_dims(mask, 0)  # Shape: (1, seq_len, seq_len)
        mask = K.cast(mask, tf.bool)
        return mask
        
    def call(self, inputs):
        #Calculate keys, queries and values vectors 
        # Calculate the vectors using the dense layer which is functionaly the same as doing a dot product
        keys = self.Wk(inputs)
        queries = self.Wq(inputs)
        values = self.Wv(inputs)

        attention_scores = tf.matmul(queries, K.transpose(keys, axes=[0,2,1]))

        # Apply the mask to the attention scores
        # Areas where the mask is true is set to -inf to ensure 0 when calculating the softmax
        masked_attention = K.where(self.mask, -np.inf, attention_scores) 

        attention_weights = K.softmax(masked_attention / self.dim**0.5, axis=-1)

        dropout_aw = self.dropout(attention_weights, training=True)
        context_vector = tf.matmul(dropout_aw, values)

        return context_vector 


# In[82]:


self_attention = CasualSelfAttention(256)
self_attention(input_embedding)

# In[83]:


class MultiHeadedAttentionWrapper(keras.Layer):
    def __init__(self, dim, num_heads=2, context_length=100, bias=True, dropout=0.5):
       super().__init__() 
       self.heads = [CasualSelfAttention(dim, context_length, bias, dropout) for _ in range(num_heads)]
    
    def call(self, inputs):
        return K.concatenate([head(inputs) for head in self.heads], axis=-1)

# In[84]:

multi_head = MultiHeadedAttentionWrapper(128)
multi_head(input_embedding)

# In[85]:

class MultiHeadAttention(keras.Layer):
    def __init__(self, config):
        super().__init__()

        # The dimention of the query, key and value weights
        self.dim = config["emb_dim"] 
        self.num_heads = config["n_heads"]
        self.bias = config["qkv_bias"]
        self.context_length = config["context_length"]
        self.head_dim = self.dim // self.num_heads
        self.dropout = keras.layers.Dropout(config["drop_rate"])
        self.mask = self.mask()

    def build(self, input_shape):
        # Initializing the query, key and value weights
        # Use the keras dense layer for the weights initialization
        self.Wq = keras.layers.Dense(self.dim, use_bias=self.bias)
        self.Wk = keras.layers.Dense(self.dim, use_bias=self.bias)
        self.Wv = keras.layers.Dense(self.dim, use_bias=self.bias)

        self.proj = keras.layers.Dense(self.dim)


    def mask(self):
        # Create & Apply a mask on the attention scores
        mask = K.triu(K.ones((self.context_length,self.context_length)), k=1) # Shape: (seq_len, seq_len)
        mask = K.expand_dims(mask, 0)  # Shape: (1, seq_len, seq_len)
        mask = K.cast(mask, tf.bool)
        return mask

    def call(self, inputs):
        
        #Calculate keys, queries and values vectors 
        # Calculate the vectors using the dense layer which is functionaly the same as doing a dot product
        keys = K.reshape(self.Wk(inputs), (-1, self.context_length, self.num_heads, self.head_dim))
        queries = K.reshape(self.Wq(inputs), (-1, self.context_length, self.num_heads, self.head_dim))
        values = K.reshape(self.Wv(inputs), (-1, self.context_length, self.num_heads, self.head_dim))
        
        keys = K.transpose(keys, axes=[0,2,1,3])
        queries = K.transpose(queries, axes=[0,2,1,3])
        values = K.transpose(values, axes=[0,2,1,3])

        attention_scores = tf.matmul(queries, K.transpose(keys, axes=[0,1,3,2]))

        # Apply the mask to the attention scores
        # Areas where the mask is true is set to -inf to ensure 0 when calculating the softmax
        masked_attention = K.where(self.mask, -np.inf, attention_scores) 

        attention_weights = K.softmax(masked_attention / self.head_dim**0.5, axis=-1)
        
        # Apply the Dropout layer
        dropout_aw = self.dropout(attention_weights, training=True)

        # Generate final context vector
        context_vector = tf.matmul(dropout_aw, values)
        context_vector = K.transpose(context_vector, axes=[0,2,1,3])
        context_vector = K.reshape(context_vector, (-1, self.context_length, self.dim))

        context_vector = self.proj(context_vector)

        return context_vector 


# In[86]:

multi_attention = MultiHeadAttention(dim=output_dim, num_heads=12, context_length=context_len, bias=False, dropout=0.3)
multi_attention(input_embedding)


# In[87]:

# Create the GPT config
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# Create a GPT model
class GPTModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.token_embedding = keras.layers.Embedding(config["vocab_size"], config["emb_dim"])

        # Create a positional embedding which has information of the word position
        self.pos_embedding = keras.layers.Embedding(config["context_length"], config["emb_dim"])
            
        # Create the dropout layer to apply to the inputs to avoid overfitting
        self.drop_emb = keras.layers.Dropout(config["drop_rate"]) 
        
        # Create the transformer blocks as a sequential model
        self.transformer_blocks = keras.Sequential( [TransformerBlock(config) 
                                                           for _ in range(config["n_layers"])])

        # Create the final normalization layer
        self.final_norm = LayerNorm(config["emb_dim"])

        self.out_head = keras.layers.Dense(config["vocab_size"])

    def call(self, inp_tokens):
        batch_size, context_len = inp_tokens.shape

        token_embedding = self.token_embedding(inp_tokens)
        pos_embedding = self.pos_embedding(tf.range(context_len))

        # Add the positional information to the original token embedding
        x = token_embedding + pos_embedding

        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        output = self.out_head(x)
        return output
        
class TransformerBlock(keras.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = LayerNorm()
        self.attention = MultiHeadAttention(config)
        self.dropout_1 = keras.layers.Dropout(config["drop_rate"])

        self.layer_norm_2 = LayerNorm()
        self.feed_forward = FeedForward(config)
        self.dropout_2 = keras.layers.Dropout(config["drop_rate"])
        
    def build(self):
        pass

    def call(self, x):     
        # Self-attention part 
        shortcut_1 = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout_1(x)
        x = shortcut_1 + x
        
        # Feed-forward part
        shortcut_2 = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = shortcut_2 + x

        return x

class LayerNorm(keras.Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def build(self, input_shape):
        emb_dim = input_shape[-1]
        self.scale = self.add_weight(shape=(emb_dim,), initializer="ones") 
        self.shift = self.add_weight(shape=(emb_dim,), initializer="zeros") 

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        var = K.var(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / K.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return 0.5 * (1 + K.tanh(K.sqrt(2/np.pi) * (x + 0.044715 * K.power(x, 3))))

class FeedForward(keras.Layer):
    def __init__(self, config):
        super().__init__()
        self.layers = keras.Sequential([keras.layers.Dense(4 * config["emb_dim"]),
            GELU(),
            keras.layers.Dense(config["emb_dim"])])

    def call(self, x):
        return self.layers(x)

model = GPTModel(GPT_CONFIG_124M)
model(next(iter(dataset))[0])
print(model.count_params())

