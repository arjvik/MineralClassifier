package com.arjvik.ironreignrobotics.mineralclassifier;

import java.io.File;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public class Main {
	private static final String FILE_PATH = "data/RoverRuckusTrainingData";
	private static final String DATA_DIRECTORY = "TrainingData";
	private static final String REPOSITORY_URL = "https://github.com/arjvik/RoverRuckusTrainingData.git";
	private static final long SEED = 6832;

	public static void main(String[] args) throws Exception {
		//Download dataset
		DataDownloader downloader = new DataDownloader();
		File rootDir = downloader.downloadFilesFromGit(REPOSITORY_URL, FILE_PATH, DATA_DIRECTORY);
		
		//Read in dataset
		DataSetIterator iterator = new CustomDataSetIterator(rootDir, 1);
		
		//Normalization
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.fit(iterator);
		iterator.setPreProcessor(scaler);
		
		//Read in test dataset
		DataSetIterator testIterator = new CustomDataSetIterator(new File(rootDir, "Test"), 1);
			
		//Test Normalization
		DataNormalization testScaler = new ImagePreProcessingScaler(0, 1);
		testScaler.fit(testIterator);
		testIterator.setPreProcessor(testScaler);
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(SEED)
				.l2(0.005)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(0, new ConvolutionLayer.Builder()
						.nIn(1)
						.kernelSize(3, 3)
						.stride(1, 1)
						.activation(Activation.RELU)
						.build())
				.build();
		
	}

}
