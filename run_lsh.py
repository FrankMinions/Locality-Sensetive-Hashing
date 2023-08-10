import csv
import jieba
import argparse
import warnings
from typing import List
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import HashingTF
from pyspark.sql import SparkSession
from pyspark.pandas import read_excel, read_csv, DataFrame
from pyspark.ml.feature import MinHashLSH

warnings.filterwarnings('ignore')

conf = SparkConf().setAppName("feature engineering").setMaster("local[*]")

# Driver heap memory size. Driver is the main control process responsible for creating context,
# submitting jobs, converting jobs into tasks, and coordinating task execution between executors.
conf.set("spark.driver.memory", '4G')

# Executor heap memory. The executor is mainly responsible for executing specific calculation tasks
# and returning the results to the driver.
conf.set("spark.executor.memory", '6G')

sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

EXCEL_FORMAT = ["xlsx", 'xls']


def readFile(file_path: str, file_format: str, is_head: bool):
    df = None
    if file_format in EXCEL_FORMAT:
        df = read_excel(file_path, header=None).values.tolist()
    if file_format == 'csv':
        df = read_csv(file_path, header=None).values.tolist()
    if file_format == 'txt':
        df = sc.textFile(file_path)

    if is_head:
        return df[1:], df[0]
    else:
        return df, None


def getTokens(df, file_format: str):

    def no_empty(s):
        if s != " ":
            return s

    lines, tokens = [], []
    if file_format == 'txt':
        for i, line in enumerate(df.collect()):
            lines.append(tuple([line]))
            token = jieba.lcut(line)
            token_ = list(filter(no_empty, token))
            tokens.append(tuple([i, line, token_]))
    else:
        for i, line in enumerate(df):
            # multiple sentences marked with token <sep>
            line_ = "<sep>".join(line)
            lines.append(tuple([i, line_]))
            token = jieba.lcut(line_)
            token_ = list(filter(no_empty, token))
            tokens.append(tuple([i, line_, token_]))

    tokenDf = spark.createDataFrame(tokens, ["id", "sequence", "tokens"])
    return lines, tokenDf


def runHashingTF(numFeatures, tokenDf):
    hashingTF = HashingTF(inputCol="tokens", outputCol="features")
    if numFeatures:
        # default to 262144 if you do not specify
        hashingTF.setNumFeatures(numFeatures)

    features = []
    rows = hashingTF.transform(tokenDf).collect()
    for row in rows:
        features.append(tuple([row.id, row.features]))

    return spark.createDataFrame(features, ["id", "features"])


def runMinHashLSH(features):
    mh = MinHashLSH()
    mh.setInputCol("features")
    mh.setOutputCol("hashes")
    model = mh.fit(features)
    return model


def getFilteredIndex(model, df, threshold):
    rows = model.approxSimilarityJoin(df, df, 0.6, distCol="JaccardDistance").collect()
    similarity, ids, filtered = [], [], []
    for row in rows:
        ids.append(row.datasetA.id)
        if row.datasetA.id != row.datasetB.id:
            if row.datasetA.id not in similarity:
                if row.JaccardDistance < threshold:
                    similarity.append(row.datasetB.id)

    for id in set(ids):
        if id not in set(similarity):
            filtered.append(id)

    return filtered


def writeFile(lines: List, index: List, file_format: str, is_head: bool, header: List, save_path: str):
    with open(save_path, mode='w', encoding='utf-8', errors='ignore') as f:
        # txt format file writing method
        if file_format == 'txt':
            for i, line in enumerate(lines):
                if i in index:
                    f.write(line[0])
                    f.write('\n')

        # csv format file writing method
        if file_format == 'csv':
            csv_writer = csv.writer(f)
            if is_head:
                csv_writer.writerow(header)
            for i, line in enumerate(lines):
                if i in index:
                    csv_writer.writerow(line[0].split('<sep>'))


def writeExcel(lines: List, index: List, is_head: bool, header: List, save_path: str):
    data = []
    for i, line in enumerate(lines):
        if i in index:
            data.append(line[0].split('<sep>'))
    if is_head:
        df = DataFrame(data, columns=header)
        df.to_excel(save_path, sheet_name='sheet1', index=False)
    else:
        df = DataFrame(data)
        df.to_excel(save_path, sheet_name='sheet1', header=False, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path of the file to be deduplicate.")
    parser.add_argument("--is_head", type=bool, default=False, help="Does the table contain a header.")
    parser.add_argument("--numFeatures", type=int, default=None, help="Using hash function to map the "
                                                                      "maximum number of features required for index mapping.")
    parser.add_argument("--threshold", type=float, default=0.5, help="The threshold of Jaccard distance.")
    parser.add_argument("--save_path", type=str, help="The path for saving filtered text.")
    args = parser.parse_args()

    if args.file_path.endswith('xlsx'):
        file_format = 'xlsx'
    elif args.file_path.endswith('xls'):
        file_format = 'xls'
    elif args.file_path.endswith('csv'):
        file_format = 'csv'
    elif args.file_path.endswith('txt'):
        file_format = 'txt'
    else:
        raise "Unsupported file format!"

    df, header = readFile(args.file_path, file_format, args.is_head)
    lines, tokenDf = getTokens(df, file_format)
    features = runHashingTF(args.numFeatures, tokenDf)
    model = runMinHashLSH(features)
    filtered = getFilteredIndex(model, features, args.threshold)
    
    if file_format not in EXCEL_FORMAT:
        writeFile(lines, filtered, file_format, args.is_head, header, args.save_path)
    else:
        writeExcel(lines, filtered, args.is_head, header, args.save_path)