[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

This repository stores simple scripts to explore scientific publication pdfs using ChatGPT API. This is created with the simple intention of sharing useful code to look across the litterature. 
THIS CODE IS EXPERIMENTAL. We share so that others could try it and if it is proves useful, please let us know. 

Running
========================

1. At the moment, there is a simple script. To run it, you first need to create your conda environment as :

```conda create --name <your_env_name> --file requirements.txt```

2. Then activate it: 

```conda activate <your_env_name>```

3. Go the script folder:

```cd scripts```

4. Copy your openAi API key in the .env file. You can find this here: https://platform.openai.com/account/api-keys

5. Run it using:

```python import_pdf_ai.py --path_pdf <path_to_your_pdf> --save_summary True```

This will save a little text file along with your pdf with the same filename but with a .txt extension. 

Parameters
========================

This script uses argparse to parse command-line arguments. Below is a summary of the available parameters:

--path_pdf: Path to a PDF file that you want to summarize.
Type: string
Default: ../example/2020.12.15.422967v4.full.pdf

--save_summary: Save the generated summary in a txt file alongside the PDF file.
Type: boolean
Default: False

--cut_bibliography: Exclude the bibliography at the end of the PDF file from the summary.
Type: boolean
Default: True

--chunk_length: Determines the final length of the summary by summarizing the document in chunks. More chunks result in a longer summary but may lead to inconsistency across sections.
Type: integer
Default: 1
Suggested values: 1 for an abstract, 2 or 3 for more detailed summaries

Example
========================
See the example/ folder for example runs. 

Credits
========================
This repository was started by Jerome Lecoq on April 12th 2023. Please reach out jeromel@alleninstitute.org for any questions. If this is useful to you, :wave: are welcome!
