# AutoLiteratureReview
Extract key research results from hundred thousands of research paper to get key insights into interested area of study. In this project, it demonstrated the use of model on COVID 19 research papers to extract profound findings about Coronavirus.

## Run Script
To run the script, go into data.py, and change filename in line 50 to file containing research paper data.
And to adjust to your interest area, change the keywords passed into data.extract_paper() to keywords that you wish to see in your literature review. For example, if you are interested in machine translation, then keywords such as 'translation' must be included.

## Research Paper Data
The research paper data must have at least doi and abstract. ('title', 'authors', 'journal', 'has_full_text' are optional, as you can change them in data.py script)

## Overall Process of Executing
The program will first generate summaries for each paper, and using unsupervised learning (clustering) to obtain most representative paper from each subset of field of study. Then produce an overall summary for your literature review.
