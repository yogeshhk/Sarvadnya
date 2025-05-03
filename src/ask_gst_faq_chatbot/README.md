# GST FAQ Bot with Rasa-NLU
Set of scripts to build a chatbot which will answer FAQ about Goods and Services Tax (GST) India.
Copyright (C) 2017 Yogesh H Kulkarni

## License:
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or any later version.

## Scripts:
* app.py: Chatbot UI built using Flask, using templates/*.html
* engine.py: Chatbot core logic as well as knowledgebase.
* config.json: Rasa NLU settings for training as well as executing intent extraction
* run_training: Windows batch file to build trained modeling
* run_server: Windows batch file to execute Rasa-NLU server.

## Dependencies:
* Needs Python 3.5

## ToDos
* Add more training data
* Entity extraction not working as desired, find out more.
* Etc.

## References
* Rasa-NLU [installation](https://github.com/RasaHQ/rasa_nlu)
* Bhavani Ravi’s event-bot [code](https://github.com/bhavaniravi/rasa-site-bot), Youtube [Video](https://www.youtube.com/watch?v=ojuq0vBIA-g)
* GST FAQs:
    * Government [Documentation](http://www.cbec.gov.in/resources//htdocs-cbec/deptt_offcr/faq-on-gst.pdf)
    * GST India [Documentation](http://www.gstindia.com/frequently-asked-questions-faqs-on-goods-and-services-tax-gst/)

## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the results of the program. It is just a fun script. Lot of improvements are still to be made. So, don’t depend on it at all.
