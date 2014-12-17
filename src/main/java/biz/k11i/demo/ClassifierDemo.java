package biz.k11i.demo;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.PrintStream;
import java.util.Random;

public class ClassifierDemo {
	public static void main(String[] args) throws Exception {
		System.out.println("\nMushroom\n");
		demo(DataSet.mushroom());

		System.out.println("\nIris\n");
		demo(DataSet.iris());
	}

	static void demo(Instances data) throws Exception {
		final int numFolds = 5;

		evaluate(new Logistic(), data, numFolds);
		evaluate(new RandomForest(), data, numFolds);
	}

	static void evaluate(Classifier classifier, Instances data, int numFolds) throws Exception {
		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(classifier, data, numFolds, new Random(123));

		PrintStream o = System.out;

		o.println("----------");
		o.printf("# %s\n", classifier.toString());
		o.printf("- Number of instances: %.4f\n", evaluation.numInstances());
		o.printf("- Weighted recall: %.4f\n", evaluation.weightedRecall());
		o.printf("- Weighted precision: %.4f\n", evaluation.weightedPrecision());
		o.printf("- Weighted AUC: %.4f\n", evaluation.weightedAreaUnderROC());
		o.printf("- Weighted f-measure: %.4f\n", evaluation.weightedFMeasure());
		o.println();

		int numClasses = data.classAttribute().numValues();
		for (int i = 0; i < numClasses; i++) {
			o.printf("## Class: %s\n", data.classAttribute().value(i));
			o.printf("- Recall: %.4f\n", evaluation.recall(i));
			o.printf("- Precision: %.4f\n", evaluation.precision(i));
			o.printf("- AUC: %.4f\n", evaluation.areaUnderROC(i));
			o.printf("- F-measure: %.4f\n", evaluation.fMeasure(i));
			o.println();
		}
	}
}
