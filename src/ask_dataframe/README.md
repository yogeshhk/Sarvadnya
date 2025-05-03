# ChatBot Interface to Data-frame
Set of scripts to build a chatbot which will query data from dataframe.
More details at https://www.kdnuggets.com/2019/06/natural-language-interface-datatable.html

Copyright (C) 2019 Yogesh H Kulkarni

## License:
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or any later version.

## Scripts:
* app.py: Chatbot UI built using Flask, using templates/*.html
* dfengine.py: Chatbot core logic as well as knowledgebase.
* cia_world_factbook.py: Web scraping CIA site
* run_training: Windows batch file to build trained modeling
* run_server: Windows batch file to execute Rasa-NLU server.


## Dependencies:
* Needs Python 3.6
* Rasa
* Spacy

## To Run:
* Execute run_training.bat
* In a separate window, execute run_server.bat
* In the original window, python app.py It will suggest a local host url, which needs to be opened in browser.

## Running

<img src="images/runningchatbot.png"/>

## ToDos
* Queries across multiple tables
* Queries with aggregation and relations, like “Which countries have more than 100 Million population and GDP per capital less than $1000?”
* Partial/Full SQL support ie to convert natural language query into an equivalent SQL query

## References
* UI: Bhavani Ravi’s event-bot [code](https://github.com/bhavaniravi/rasa-site-bot), Youtube [Video](https://www.youtube.com/watch?v=ojuq0vBIA-g)
* Data gathering: Web Scraping with Python: Illustration with CIA World Factbook https://www.kdnuggets.com/2018/03/web-scraping-python-cia-world-factbook.html


## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the results of the program. It is just a fun script. Lot of improvements are still to be made. So, don’t depend on it at all.
