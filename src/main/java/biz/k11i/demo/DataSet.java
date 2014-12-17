package biz.k11i.demo;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class DataSet {
	private static final File tempBaseDir = new File("./dataset");

	public static Instances abalone() throws IOException {
		String url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data";
		File file = download(url);

		CsvLoaderWithoutHeader loader = new CsvLoaderWithoutHeader();
		loader.setSource(file);
		loader.setNominalAttributes("first");

		loader.setStructure("abalone",
				"Sex",
				"Length",
				"Diameter",
				"Height",
				"Whole weight",
				"Shucked weight",
				"Viscera weight",
				"Shell weight",
				"Rings");

		Instances result = loader.getDataSet();
		result.setClassIndex(8);

		return result;
	}

	public static Instances iris() throws IOException {
		String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
		File file = download(url);

		CsvLoaderWithoutHeader loader = new CsvLoaderWithoutHeader();
		loader.setSource(file);
		loader.setNominalAttributes("last");
		loader.setStructure("iris",
				"sepal length",
				"sepal width",
				"petal length",
				"petal width",
				"class");

		Instances result = loader.getDataSet();
		result.setClassIndex(4);

		return result;
	}

	public static Instances mushroom() throws IOException {
		String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data";
		File file = download(url);

		CsvLoaderWithoutHeader loader = new CsvLoaderWithoutHeader();
		loader.setSource(file);
		loader.setNominalAttributes("first-last");
		loader.setStructure("mushroom",
				"class",

				"cap-shape",
				"cap-surface",
				"cap-color",
				"bruises?",
				"odor",

				"gill-attachment",
				"gill-spacing",
				"gill-size",
				"gill-color",
				"stalk-shape",

				"stalk-root",
				"stalk-surface-above-ring",
				"stalk-surface-below-ring",
				"stalk-color-above-ring",
				"stalk-color-below-ring",

				"veil-type",
				"veil-color",
				"ring-number",
				"ring-type",
				"spore-print-color",

				"population",
				"habitat");

		Instances result = loader.getDataSet();
		result.setClassIndex(0);
		return result;
	}

	public static Instances scale(Instances input) throws Exception {
		Normalize normalize = new Normalize();
		normalize.setInputFormat(input);
		return Filter.useFilter(input, normalize);
	}

	private static File download(String url) {
		String[] elems = url.split("/");
		String filename = elems[elems.length - 1];
		File tempFile = tempBaseDir.toPath().resolve(filename).toFile();

		if (tempFile.exists()) {
			return tempFile;
		}

		URL _url;
		try {
			_url = new URL(url);
		} catch (MalformedURLException e) {
			throw new RuntimeException("Malformed URL: " + url, e);
		}

		HttpURLConnection conn;
		try {
			conn = (HttpURLConnection) _url.openConnection();
		} catch (IOException e) {
			throw new RuntimeException("Cannot open URL connection: " + url, e);
		}

		if (!tempBaseDir.exists()) {
			if (!tempBaseDir.mkdirs()) {
				throw new RuntimeException("Cannot make temporary directory: " + tempBaseDir.toString());
			}
		}

		try {
			conn.setConnectTimeout(1000 * 10);
			conn.setReadTimeout(1000 * 10);

			try {
				tempFile.createNewFile();

			} catch (IOException e) {
				throw new RuntimeException("Cannot create temporary file: " + tempFile, e);
			}

			try {
				Files.copy(conn.getInputStream(), tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

			} catch (IOException e) {
				throw new RuntimeException("I/O error occurred", e);
			}

			return tempFile;

		} finally {
			conn.disconnect();
		}
	}

	static class CsvLoaderWithoutHeader extends CSVLoader {
		void setStructure(String relationName, String... columnNames) {
			FastVector attribNames = new FastVector();

			for (String header : columnNames) {
				attribNames.addElement(new Attribute(header, (FastVector) null));
			}

			m_structure = new Instances(relationName, attribNames, 0);
		}
	}
}
