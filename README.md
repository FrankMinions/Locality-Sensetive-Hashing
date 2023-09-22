# Locality-Sensetive-Hashing
Based on pyspark, the locality sensetive hashing (LSH) algorithm code is implemented to process massive text in the scene of deduplication. Many academic works regard this as a necessary step in feature engineering. Due to the fact that the underlying code is based on Spark, it is necessary to install Java in advance. Relevant documents can be referenced at https://spark.apache.org/docs/latest/api/python/index.html.

In view of the need to support Chinese and English languages under normal circumstances, for this consideration, I chose the tokenizer of Baichuan2-13B-Base as the segmentation tool.

Note that sentences are separated by `<unk>`.