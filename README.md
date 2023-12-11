# Classifier

## Call TFIDFClassifier function

### Exporting all files
Please place the files and TFIDFClassifier.sav into a directory call Classifier.
you may require to extract TFIDFClassifier.sav from the zip file.
The following files needed:
- TFIDFCLassifier.py
- TFIDFClassifier.sav
- test_data_Processed.csv
- train_data_Processed.csv

#### Install requirements
> pip install -r requirements.txt
For the very first time you need to uncomment the ntlk to download the stopwords and the two other.

### Import function
```text
from Classifier.TFIDFClassifier import classifierByTFIDF
```

### Call function
```bash
# Returns "Positive", "Neutral" or "Negative"
Result = classifierByTFIDF("INSERT YOUR REVIEW HERE")
```


