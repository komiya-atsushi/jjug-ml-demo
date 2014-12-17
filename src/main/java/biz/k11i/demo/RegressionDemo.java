package biz.k11i.demo;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

import java.io.PrintStream;
import java.util.Random;

public class RegressionDemo {
	public static void main(String[] args) throws Exception {
		Instances abalone = DataSet.abalone();

		final int numFolds = 5;

		evaluate(new LinearRegression(), 1e-5, abalone, numFolds);
	}

	static void evaluate(LinearRegression regression, double ridge, Instances data, int numFolds) throws Exception {
		regression.setRidge(ridge);

		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(regression, data, numFolds, new Random(123));

		PrintStream o = System.out;

		o.println("----------");
		o.printf("# %s\n", regression.toString());
		o.printf("- Correlation coefficient: %.4f\n", evaluation.correlationCoefficient());
		o.printf("- Mean absolute error: %.4f\n", evaluation.meanAbsoluteError());
		o.printf("- Root mean squared error: %.4f\n", evaluation.rootMeanSquaredError());
		o.println();
	}
}
