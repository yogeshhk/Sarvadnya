# Notes on ChatBot Interface to Data-frame/base
Set of scripts to build a chatbot which will query data from data-base/frame.

Copyright (C) 2019 Yogesh H Kulkarni

## Thoughts
Structured data access would be less laborious if a Chatbot interfaces is provided to it. Building a simple chatbot here, to access fields from csv dataframe.
This is a start as it can be enhanced further to have more EDA (Exploratory Data Analysis) capabilities. Even more would be if SQL query can be generated from users input and response can be fetched from a Database.

## Sample Domain
*	Dataframe is populated from CIA World Fact book dataset
*   Tutorial for scraping referred is at https://www.kdnuggets.com/2018/03/web-scraping-python-cia-world-factbook.html

## HTML Parsing

    HTML:
        a tags are links, The href property of the tag determines where the link goes.
        div – indicates a division, or area, of the page.
        One element can have multiple classes, and a class can be shared between elements.
        Each element can only have one id, and an id can only be used once on a page. 
        Classes and ids are optional, and not all elements will have them.
        adding classes and ids doesn’t change how the tags are rendered at all.
        They are used as MARKERS.
        Classes and ids are used by CSS to determine which HTML elements to apply 
        certain styles to. We can also use them when scraping to specify specific
        elements we want to scrape. 
    
    Data:
        <li>
            <h2 ccode="af" sectiontitle="Geography" ...>
            Geography ::
            <span class="region">AFGHANISTAN</span>
            </h2>
        </li>
        <li>
            <div id="field">
            <a >Location</a>
            Location:
            <div class="category_data">
            Southern Asia, north and west of Pakistan, east of Iran
            </div>
            :
            :
        </li>
        <li>
            <h2 ccode="af" class="question sas_med" sectiontitle="People and Society">
            People and Society ::
            <span class="region">
             AFGHANISTAN
            </span>
           </h2>
        </li>
        <li>
            <div class="category sas_light" id="field" >
                <a >Population:</a>
            </div>
            <div class="category_data">
                33,332,025 (July 2016 est.)
            </div>
            :
            :
            <div class="category sas_light" id="field">
                <a>Languages:</a>
                <a href="../fields/2098.html#af"><img src=/></a>
            </div>
            <div class="category_data">
                Afghan Persian or Dari (official) 50%, Pashto (official) 35%, Turkic languages (primarily Uzbek and Turkmen) 11%, 30 minor languages (primarily Balochi and Pashai) 4%, much bilingualism, but Dari functions as the lingua franca
            </div>
            <div>
                <span class="category">
                    note:
                </span>
                <span class="category_data">
                    the Turkic languages Uzbek and Turkmen, as well as Balochi, Pashai, Nuristani, and Pamiri are the third official languages in areas where the majority speaks them
                </span>
            </div>

        </li>
'''

'''
    Two formats for class having "category": span and div
    div has "sas_light" also in the class name.
    e.g
    Map references:
        Asia
    span has no other word
    for span, we need another dictionary to collect the pairs
    e.g.
    Area:
        total: 652,230 sq km
        land: 652,230 sq km
        water: 0 sq km
'''


## To Do
*	Evaluate
	* Sent2SQL : Richard Sochar
	*
	
* 	Implement sentence embeddings

## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the results of the program. It is just a fun script. Lot of improvements are still to be made. So, don’t depend on it at all.
