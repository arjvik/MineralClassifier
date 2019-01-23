package com.arjvik.ironreignrobotics.mineralclassifier;

import java.io.File;
import java.io.IOException;

public class DataDownloader {
	
	public DataDownloader() {
		
	}
	
	public File downloadFilesFromGit(String repositoryUrl, String filePath, String dataDirectory) throws IOException, InterruptedException {
		if(new File("data/RoverRuckusTrainingData").exists()) {
			System.out.println("Training Data already exists, updating to latest version (via git)");
			Process p = new ProcessBuilder(
								String.format("git -C %s pull", filePath)
								.split(" "))
							.start();
			p.getInputStream().transferTo(System.out);
			p.getErrorStream().transferTo(System.out);
			p.waitFor();
			System.out.println("Successfully updated training data");
		} else{
			System.out.println("Training Data not available, downloading");
			Process p = new ProcessBuilder(
								String.format("git clone --depth=1 %s %s", repositoryUrl, filePath)
								.split(" "))
							.start();
			p.getInputStream().transferTo(System.out);
			p.getErrorStream().transferTo(System.out);
			p.waitFor();
			System.out.println("Successfully downloaded training data");
		}
		return new File(new File(filePath), dataDirectory);
	}
}
