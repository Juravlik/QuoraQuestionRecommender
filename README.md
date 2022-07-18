## Question recommender system, based on Faiss index and KNRM ranker
***Notebook [examples.ipynb](examples.ipynb) contains examples of using Flask api***

### 1. Faiss index part:
For the input question the **30** closest questions (by euclidian distance) are searched in the Faiss index. 

### 2. KNRM ranker:
**30** nearest neighbors come to the entrance of the pre-trained KNRM model, after reranking the top-10 similar questions are sent as a response.

**_Note:_** You can change number of questions that Faiss and KNRM return in [app_config](configs/app_config.json)


#### KNRM model architecture:
![knrm_architecture](readme_utils/knrm_architecture.png)
![model_description](readme_utils/model_description.png)

### How to use:
##### 1. Install requirements:
  - clone repo
  - download both folders from [google drive](https://drive.google.com/drive/folders/1rQoE-CklSySBZP9pGUjvabb3SVWjXxAB?usp=sharing) to repo folder
  - `$ pip install -r requirements.txt`
##### 2*. Train model (it takes about 10 min, also it is not necessarily, pretrained weights are already in weights folder)
- `$ python train_model.py` in case if you want to train model by yourself (it takes about 10 min). Also you can change some parameters in [trainer_config](configs/trainer_config.json)
##### 3. Run flask server:
- `$ python app.py`

##### 4. `/ping` until status "ok"

##### 5. `/update_index` to add your questions to Faiss index

##### 6. `/query`
**query structure:**
```python
{
  "queries" : ["How many stars are in the sky?", ..., "What is the easiest language?"]
}
```
##### 7. return:
**answer structure:**
```python
{
  "lang_check" : [True, ..., False], # if eng language was detected
  "suggestions" : [
      [[<question id in index>, <similar question>], ..., [<question id in index>, <similar question>]], # suggestion ordered by relevance from most relevant to least
      ...
      [[<question id in index>, <similar question>], ..., [<question id in index>, <similar question>]]
   ]
}
```

##### example:
query:
```python
{'queries': ['Проверка на язык',
             'Do you know something about Hillary Clinton?']}
```
answer:
```python
{
    "lang_check": [
        false,
        true
    ],
    "suggestions": [
        null,
        [
            [
                "45068", "Do you dislike Hilary Clinton? If so, please could you explain why?"
            ],
            [
                "102609", "Why do people dislike Hillary Clinton? What has she done wrong?"
            ],
            [
                "326387", "What do you most admire about Hillary Clinton?"
            ],
            [
                "45544", "Do you think Hillary Clinton will win?"
            ],
            [
                "258242", "What do you think about Memrise?"
            ],
            [
                "261490", "What do you think about AliExpress?"
            ],
            [
                "241886", "What's something you wish people knew about you?"
            ],
            [
                "425857", "What do you think about Stoicism?"
            ],
            [
                "222146", "What do you really know about Iran?"
            ],
            [
                "189696", "Why does everyone hate Hillary Clinton?"
            ]
        ]
    ]
}
```
