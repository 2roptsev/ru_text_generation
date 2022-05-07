# News generation

Neural network for text generation trained on news domain (lenta.ru).
Нейросеть, генерирующая новости на русском языке. Обучена на заголовках сайта lenta.ru.
[dataset](https://drive.google.com/open?id=1NlFuOjOt0oQ9Mx70Z7ZvfOsB3-1fCALp). 

- Pytorch
- LSTM RNN model
- Different token levels: word-level tokens (spacy), char-level, byte-pair encoding

## Features

- Allows to train text generation and try different tokens levels
- Different generators: argmax, distributional, beam search
- Train or finetune on any text file

Авторы репозиториев текстовой генерации на русском языке используют токены отдельных слов, получая несвязные предложения, так как существует проблема обратной лемматизации и связи слов в предложении. Использование Char-RNN не позволяет добиться хороших результатов генерации в связи с угасанием градиентов, так как последовательность из отдельных символов имеет значительно большую длину. Использование byte pair encoding позволяет добиться реалистичных результатов на небольшом датасете (~60к предложений) без использования предобученных моделей:
> сми сообщили о планах сша оставить россиян на заседание совбеза оон

> власти подмосковья раскрыли возможность распространения мигрантов

> сми сообщили о нежелании роналду пропустить в россии собственное время

> в минэкономразвития опровергли рост ввп россии на 50 процентов

> в кремле отказались комментировать информацию об освобождении от сша

Authors of different repositories on text generation on russian language use only word-level tokens, wich leads to non-connected words or Char-RNN which leads to low quality generation. Byte pair encoding approach used here leads to realistic results on small (60k sentences) dataset from scratch.

## Try

After installing requirements.txt launch python3 in repo directory:

```
>>> from init_generator import get_stated_generator
>>> sg = get_stated_generator()
>>> sg.roll()
'сми сообщили о планах сша оставить россиян на заседание совбеза оон'
```

## Train

Train examples could be found in [prepare_data_and_train_experiments.ipynb][notebook].

To train on your data, prepare it using notebook, than run [train.py][train].

```
> python3 train.py --help
usage: train.py [-h] [--data-path DATA_PATH] [--embedding-dim EMBEDDING_DIM]
                [--hidden-dim HIDDEN_DIM] [--device-name DEVICE_NAME]
                [--n-epochs N_EPOCHS] [--batch-size BATCH_SIZE]
                [--checkpoint-path CHECKPOINT_PATH]

Train RNN model for generation.

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        path to file train sentences
  --embedding-dim EMBEDDING_DIM
                        embedding size
  --hidden-dim HIDDEN_DIM
                        hidden state size
  --device-name DEVICE_NAME
                        device name. to use cpu for training change value to
                        any
  --n-epochs N_EPOCHS
  --batch-size BATCH_SIZE
  --checkpoint-path CHECKPOINT_PATH
                        path to save trained model
```

## License

MIT

   [notebook]: <https://github.com/2roptsev/ru_text_generation/tree/master/prepare_data_and_train_experiments.ipynb>
   [train]: <https://github.com/2roptsev/ru_text_generation/tree/master/train.py>
