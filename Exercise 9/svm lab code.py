from ast import parse
from pyspark import SparkContext
from pyspark.sql import SparkSession
# $example on$
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
# $example off$
if __name__ == "__main__":
        sc = SparkContext(appName="PythonSVMWithSGDExample")
        def parsePoint(line):
                values = [float(x) for x in line.split(' ')]
                return LabeledPoint(values[0], values[1:])


        data = sc.textFile("D:\\MEPCO\\SEMESTER 5\\Machine Learning Essentials\\Exercise 9\\Raisin_Dataset.txt")
        parsedData = data.map(parsePoint)

        # Build the model
        model = SVMWithSGD.train(parsedData, iterations=100)

        # Evaluating the model on training data
        labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
        trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
        print("Training Error = " + str(trainErr))

        trainAcc = labelsAndPreds.filter(lambda lp: lp[0] == lp[1]).count() / float(parsedData.count()) 
        print("Accuracy = ",trainAcc)

        # # Save and load model
        # model.save(sc, "pythonSVMWithSGDModel")
        # sameModel = SVMModel.load(sc, "pythonSVMWithSGDModel")
        # # $example off$
        # trainAcc = labelsAndPreds.filter(lambda lp: lp[0] == lp[1]).count() / float(parsedData.count()) 
        # print("Accuracy = ",trainAcc)