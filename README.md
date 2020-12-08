# paintings-generator
To install all packages, run the following command:
	
		pip3 install requirements.txt

To scrape images, run the following command:

		python3 scrape_images.py

Adjust the inputs for the for loop in line 18 to handle how many entries are being scraped. 
Scraped images will be saved to the ‘scraped_images/’ folder within the project. 
To resize, sharpen, and flip images, run the following command:
		
		python3 preprocessing.py

This script will resize all the scraped images, sharpen, and flip them, and then save them to ‘dataset/Images.’

To run the model for the GAN, open the notebook titled ‘train_images.ipynb’.
Run all of the cells. The last cell will allow for the viewing of the generated images for a specified epoch. The chosen epoch can be changed in that cell. 
The resulting pkl file will be saved as ‘train_samples.pkl.’ 

To calculate FID Scores, open the notebook ‘calculate_FID.ipynb’ and run every cell in that notebook. The default pkl object being used is the ‘train_samples.pkl’ file. 
The FID scores are then calculated for each epoch used in the training and the average and standard deviations are reported. 

