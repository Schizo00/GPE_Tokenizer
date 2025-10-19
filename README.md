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

## Based on the work:

```
@inproceedings{velayuthan-sarveswaran-2025-egalitarian,
 title = "Egalitarian Language Representation in Language Models: It All Begins with Tokenizers",
 author = "Velayuthan, Menan and
 Sarveswaran, Kengatharaiyer",
 editor = "Rambow, Owen and
 Wanner, Leo and
 Apidianaki, Marianna and
 Al-Khalifa, Hend and
 Eugenio, Barbara Di and
 Schockaert, Steven",
 booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
 month = jan,
 year = "2025",
 address = "Abu Dhabi, UAE",
 publisher = "Association for Computational Linguistics",
 url = "https://aclanthology.org/2025.coling-main.400/",
 pages = "5987--5996",
}
```
