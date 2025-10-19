## Installation
#### ```pip install gpe-tokenizer```

## Basic Usage
#### ```from gpe_tokenizer import SinhalaGPETokenizer```

### Model Compatibility
#### For BERT
#### ```tokenizer = SinhalaGPETokenizer(model='bert')```

#### For llama
#### ```tokenizer = SinhalaGPETokenizer(model='llama')```

#### For GPT
#### ```tokenizer = SinhalaGPETokenizer(model='gpt')```

### Tokenize
#### ```tokenizer.tokenize(text)```



## Tokenizer Training Details
#### Corpus Size: 10 Million Sentences
#### Vocab Size: 32000
#### Training Time: 13H 29M