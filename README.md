# Vector Search API
Vector-Search API of databases.

## Quick Start
1. Install package.
```sh
(env) pip install vector-search-api
```

2. Execute code.
```python
>>> from vector_search_api.searcher import InMemoryVectorSearch
>>>
>>> vs_api = InMemoryVectorSearch(project_name='vs')
>>>
>>> documents = [{
...     'text': 'You good.',
...     'metadata': {'meta': 1},
...     'vector': [-0.034,  0.579, -0.587, -0.777, -0.753, -0.425, -0.607,  0.042],
... }, {
...     'text': 'I good.',
...     'metadata': {'meta': 2},
...     'vector': [0.274, -0.938,  0.041,  0.070, -0.790, -0.439,  0.585,  0.288],
... }, {
...     'text': 'Every one good.',
...     'metadata': {'meta': 3},
...     'vector': [0.881, -0.409, -0.164, -0.922,  0.771, -0.892, -0.433, -0.307],
... }]
>>> vs_api.insert_documents(documents)
>>>
>>> vs_api.search_documents(
...     [0.293 , -0.129,  0.969 ,  0.077, -0.219, 0.274,  0.721,  0.195]
... )
[{'metadata': {'meta': 2},
  'similarity': 0.39819718992710423,
  'text': 'I good.',
  'vector': array([ 0.274, -0.938,  0.041,  0.07 , -0.79 , -0.439,  0.585,  0.288])},
 {'metadata': {'meta': 3},
  'similarity': -0.2866921349941669,
  'text': 'Every one good.',
  'vector': array([ 0.881, -0.409, -0.164, -0.922,  0.771, -0.892, -0.433, -0.307])},
 {'metadata': {'meta': 1},
  'similarity': -0.53701636780471,
  'text': 'You good.',
  'vector': array([-0.034,  0.579, -0.587, -0.777, -0.753, -0.425, -0.607,  0.042])}]
```

## Basic Usages

## Roadmap
- [x] InMemoryVectorSearch
- [ ] Batch encoder.
- [ ] ElasticSevenVectorSearch
- [ ] ElasticVectorSearch
- [ ] OpensearchVectorSearch
- [ ] PineconeVectorSearch
