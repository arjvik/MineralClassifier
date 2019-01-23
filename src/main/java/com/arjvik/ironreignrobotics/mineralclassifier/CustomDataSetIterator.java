package com.arjvik.ironreignrobotics.mineralclassifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.stream.Stream;

import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class CustomDataSetIterator extends IteratorDataSetIterator implements DataSetIterator {

	private static final int ROW_STEP = 8,
							 COL_STEP = 8;
	private static final long serialVersionUID = 1L;

	public CustomDataSetIterator(File rootDir, int batchSize) {
		super(getIterator(rootDir), batchSize);
	}

	private static Iterator<DataSet> getIterator(File rootDir) {
		File[] files = rootDir.listFiles((file, name) -> file.isFile() &&
														 name.endsWith(".txt") &&
														 name.startsWith("LabledMineral-"));
		return Stream.of(files)
					 .map(CustomDataSetIterator::fileToDataSet)
					 .iterator();
	}
	
	
	private static DataSet fileToDataSet(File inputFile) {
		try (FileReader fileReader = new FileReader(inputFile);
				 BufferedReader reader = new BufferedReader(fileReader)) {
			String[] coordLine = reader.readLine().split(",");
			INDArray output = Nd4j.create(new double[]{Double.parseDouble(coordLine[0]),
													  Double.parseDouble(coordLine[1])});
			
//			int[][][] inputArray = new int[240/ROW_STEP][320/COL_STEP][]; 
			double[][] inputArray = new double[240/ROW_STEP][3*320/COL_STEP];
			
			for (int r = 0; r < 240; r+= ROW_STEP) {
				String[] line = reader.readLine().split(",");
				for (int c = 0; c < 320; c+= COL_STEP) {
					int red = Integer.parseInt(line[3*c]);
					int green = Integer.parseInt(line[3*c+1]);
					int blue = Integer.parseInt(line[3*c+2]);
					
					//inputArray[r][c] = new int[]{red, green, blue};
					inputArray[r][3*c]   = red;
					inputArray[r][3*c+1] = green;
					inputArray[r][3*c+2] = blue;
				}
			}
			INDArray input = Nd4j.create(inputArray);
			return new DataSet(input, output);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
}