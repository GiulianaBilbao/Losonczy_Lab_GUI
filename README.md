# Losonczy_Lab_GUI
Labeling GUI for checking classification of spikes in multiple ROIs

**GENERAL NOTES:**

0) Your current environments should be fine but you may need to _pip install dashly_
	- There shouldn't be any other additional installs but I included the environment file I used in the google drive just in case.

1) Running 01_22_2025_GUI_w_ROI in VScode or from a terminal will show a URL in the terminal

2) Opening the URL in a web browser will let you look at the GUI

3) You will need to upload the data once you open the GUI
	- I included a formatingPreviousDataForConfig.ipynb to show you how I formatted the data. It makes it into a .JSON file although it should be a .pkl file because it's meant to just show the formatting
	- The GUI will accept the .JSON or .pkl files
	- In the Google Drive folder I included a formatted .pkl file

4) You can keep editing without reloading the visualizer but the reload button will only update the visualizer of the plot you're currently working on
	- However! The data will get updated even if it doesn't appear in the visualizer
	- If you go back to editing that ROI and reload it you will see any previous points you added

5) To draw an event hover over the plot and you will see a little dashed square in the top right corner
	- This is called "box select", it will let you draw the square - you should see an update regarding what data is within the boxed region after you try this

6) To add or remove an event you have to explicitly click those buttons
	- You won't visually see the point added until you click to reload the visualizer
	- Once you add or remove this automatically updates your dataset

7) The files will export as a .pkl file in the same architecture as in the jupyter notebook if you would like to see the formatting!
   - If you quickly want to convert fromthe .JSON format in the jupyter notebook to the .pkl format just run it through the GUI and export it (it'll covert it)
