# Sequence Labeling for Reference Parsing of Cyrillic Script Scholarly Data

## Task
- Generate a large data set of labeled Cyrillic reference strings, consisting of over 100,000 synthetically generated references as well as over 700 references manually labeled references gathered from multidisciplinary Cyrillic script publications
- Fine-tune a multilingual BERT model on different data set sizes and determine the amount of training data needed to train high-performance deep learning models (BERT in our case) for Citation Field Extraction (CFE) and assess the benefits of training on synthetically generated references. We achieve a F1-Score of 0.933 with our best BERT model

## Content 
Below the structure of contents (code) and their descriptions are summarized. The paths refer to the data folder structure. The data used for training and evaluation (>100,000 synthetically generated & manually labeled references) can be found here: https://figshare.com/s/50f02a32af7447b2fdbc.
Under the header "Data" additional information to the contents and folder structure of our data is provided.

# Code

## Synthetic-Data-Set
### WOS2Excel

Description: Transformation of raw WoS data to Excel files

**wos_article_data_to_excel.ipynb**: Transformation of  “Article”-type WoS raw data to Excel files

* Input: WoS raw data 
* Output: Excel file

**wos_meeting_data_to_excel.ipynb**: Transformation of  “Conference Proceeding (meeting)”-type WoS raw data to Excel files 

* Input: WoS raw data 
* Output: Excel file

### Bib2PDF
Description: Generate a PDF and BibTeX file per citation style (**XX**) and reference

**process_one_citation_XX.py**:

* Input: 
     * --filename: big BibTeX filename (34K_final_v3.bib)
     * --path: path of BibTeX and output (Folder where BibTeX and BST is located and the output will be saved)

* Output:
One BibTeX and PDF file per reference (“.\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\pdf_bibtex_data”)

### PDF2labeled_text

Description: Generate a XML file (labeled text) per citation style (**XX**) and reference

**XX_synthetic_pdf_to_labelled_text-distance_algo-fine_grained.ipynb**:

* Input: BibTeX and PDF files
* Output: Labeled text (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\Labeled_text_data")

### Statistics_synthetic_dataset.ipynb

Description: Statistics of the synthetic data set

More statistics in: “.\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\Synthetic_data_statistics”

## Manuall-Annotated-Data-Set

### XML11_to_TEI.ipynb

Description: Transformation of Inception XML1.1 files to TEI files incl. header labels

* Input: Inception XML1.1 files (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-
Data\Real_annotated_data\manual_anno_xml1.1")

* Output: TEI (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\TEI")

### Inception_labelled_text.ipynb

Description: Transformation of Inception XML1.1 files to labeled reference strings

* Input: Inception XML1.1 files (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\manual_anno_xml1.1")

* Output: Labeled text (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\labelled_text_per_paper”)

### Inception_labelled_text-language.ipynb

Description: Transformation of Inception XML1.1 files to labeled reference strings per language

* Input: Inception XML1.1 files (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\manual_anno_xml1.1")

* Output: Labeled Text (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\labelled_text_per_language”)

### Statistics_manual_dataset.ipynb

Description: Statistics of the manually annotated data set


## Approach

### Grobid

**grobid_create_test_set.ipynb**: Transformation of the manually annotated data set to GROBIDs TEI format files

* Input: Labeled text test set (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\labelled_text_per_paper")

* Output: Grobid TEI (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Grobid\grobid_test_data")

**grobid_create_training_data_token.py**: Transformation of the synthetic data set to GROBIDs TEI format files

* Input: Labeled text train set (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\Labeled_text_data")

* Output: Grobid TEI (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Grobid\grobid_training_data")

**grobid_copy_sample_to_corpus.py**: A random sample of the training data is created and copied to the designated GROBID folder. Depending on the number of data that is aimed to be used for training, the random sample number has to be adjusted.

Here: ./grobid-0.6.1/grobid-trainer/resources/dataset/citation/evaluation/" the test data set should be stored (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Grobid\grobid_test_data").

**Grobid_train_eval.txt**: Batch commands for training and evaluation

### BERT
#### Training_Evaluation_synthetic_hold_out_set

Description: Training of BERT models and evaluation on synthetic and manually annotated data.

**create_test_train_set.py**: Data set split. Hold-out test set: 2,000 synthetic references (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\test_set_synth”) Training data set: remaining synthetic references (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\train_set_synth”)

**create_train_all.py**: Generate random samples from training data set for training of the different BERT models with varying training data set sizes (**XX**). 

* Input: Labeled reference strings train data (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\train_set_synth”)

* Output: Random samples (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\train_random_samples\train_**XX**”)

**Folder Test/Train**: Training and evaluation of the BERT models.

So created BERT models are stored here: „.\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\models“.

Results of evaluation and training durations are stored here: „.\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\Evaluation_final“.


#### Training_Evaluation_avg_over_5_models
Description: Per training data set size (**XX**) 5-times different random samples are created and 5 models are trained.

**bert_train_eval_5avg_XX.py**:

* Input: Labeled reference strings train data (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\train_set_synth”)

* Output: evaluation metrices in .txt (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_results_real_set”)

#### Final_model_2K

Final BERT model trained on 2,000 references

**bert_train_final.py**: With different random samples of 2,000 references 5 BERT models are trained. We selected the second model as our final BERT model, that will be evaluated in the subsequent scripts. The final model is stored here: (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model”)

**bert_test_final.py**: The final BERT model is evaluated on synthetic and manually annotated test data. Results are stored here: (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model\eval_final_2.txt”)

**bert_test_final_language.py**: Evaluation oft he final model on the manually annotated data per language.

* Input: Model & Tag_values(".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model”) and “real” reference data per language (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\labelled_text_per_language”)

* Output: Evaluation results; eval_englisch.txt, eval_other.txt,  eval_russian.txt, eval_ukrainian.txt (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model”)

**bert_test_final_confusion_matrix.py**: The final BERT model is evaluated on real data and the confusion matrix is calculated. Results are stored here: (".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model\confusionmatrix.txt”)

# Data
Structure of contents (data) and their descriptions.

## BERT

### train_random_samples
Training data for BERT models with differing training data.

### BERT_final_model
* Final BERT model  [model_2.pth, tag_values_2] trained with 2,000 instances
* Evaluation of the model on annotated data per [eval_english, eval_other, eval_ukrainian, eval_russian]
* Evaluation of the model on synthetic hold-out dataset and real annotated [eval_final_2.txt] and confusion matrix of the results on real annotated data [confusionsmatrix.txt]

### BERT_results_real_set
Results of all BERT models (Five iterations per training data size) evaluated on the real annotated data

### Evaluation_final
Results of BERT models trained with differing training dataset sizes, evaluated on the real annotated data and the synthetic hold-out dataset. Additionally, files with duration of model training per dataset size.

### models
One-time trained models on differing dataset sizes. Used for evaluations in „Evaluation_final“.

## FastText
FastText embeddings. Used to determine the languages of the manually annotated references.

## Grobid
### Grobid_training_data
Synthetically created labelled reference strings in GROBID´s TEI format.

### Grobid_test_data
Manually created labelled reference strings in GROBID´s TEI format.

### Grobid_results
Evaluation of GROBID models (differing training dataset sizes).

## Real_annotated_data
### CORE_pdfs
PDFs that have been manually annotated and which resulted in the „real“ labeled reference string dataset.
### Manual_anno_xml1.1
Annotated data exported from Inception in XML1.1 format. 
### TEI
Annotated data transformed into TEI files (TEI files have the same structure, as the TEI output files of GROBID)
### Labelled_text_per_paper
Labelled reference strings per annotated paper. 
### Labelled_text_per_language
Labelled reference strings per language

## Synthetic_data 

### Big_BibTeX
BibTeX file that contain all paper metadata gathered from WoS. Exported from EndNote.

### BST files
Citation style files.

### EndNote_final_library
EndNote library with all paper metadata from WoS.

### Labeled_text_data
All labelled synthetic reference strings

### Pdf_bibtex_data
All PDFs and BibTeX files per reference and citation style (automatically generated with python).

### Synthetic_data_statistics
Synthetic dataset statistics.

### Test_set_synth
Hold-out-set of 2,000 labeled synthetic reference strings

### Train_set_synth
Labelled synthetic reference strings without the hold-out set.

### WoS_Data
Raw data from WoS and in Excel format.
