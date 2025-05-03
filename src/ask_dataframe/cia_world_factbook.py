'''
    Project: CIA WorldFactbook Web scraping
    Site: https://www.cia.gov/library/publications/the-world-factbook/print/textversion.html
    Collecting: Country-wise data
    Author: yogeshkulkarni@yahoo.com

    World Factbook is presented at the site in field wise manner
        Languages: https://www.cia.gov/library/publications/the-world-factbook/fields/2098.html#135
    It is also presented in country wise manner
        Afganistan: https://www.cia.gov/library/publications/the-world-factbook/geos/print_af.html
	
	Ref: https://www.kdnuggets.com/2018/03/web-scraping-python-cia-world-factbook.html

'''
import requests
from bs4 import BeautifulSoup, element
from collections import OrderedDict
import json
import pandas as pd
import csv
def collect_country_wise_information():
    # country_code_dict = {'Afghanistan': 'af', 'Argentina': 'ar'}
    country_code_dict = ccodes()
    # country_code_dict = {'Afghanistan': 'af'}
    master_dict = OrderedDict()
    for k,v in country_code_dict.items():
        url = 'https://www.cia.gov/library/publications/the-world-factbook/geos/print_' + v + '.html'
        country_info_dict = collect_single_country_information(url)
        if country_info_dict is None:
            continue
        master_dict[k] = country_info_dict
    return master_dict

def ccodes():
    with open('data/ccodes.csv', mode='r') as infile:
        reader = csv.reader(infile)
        mydict = OrderedDict()
        for rows in reader:
            mydict[rows[0]] = rows[1].lower()
        return mydict

def get_page_object(page_url):
    page = requests.get(page_url)
    if page.status_code != 200:
        print("{} was NOT downloaded successfully".format(page_url))
        return None
    soup = BeautifulSoup(page.content, 'html.parser')
    # print(soup.prettify())
    return soup


def collect_single_country_information(print_info_url):
    print("Collecting infor for {}".format(print_info_url))
    page_obj = get_page_object(print_info_url)
    if page_obj is None:
        return None

    # we can move through the structure one level at a time. Using 'children'
    # children returns a list generator, so we need to call the list function on it
    # 'Tag' or actual content is in the 4th object ['\n', 'doctype html', '\n', <html>..]
    html = list(page_obj.children)[3]
    # there are two tags here, head, and body, need to go into body
    body = list(html.children)[3]
    content = list(body.children)[1]
    article = list(content.children)[1]
    text = list(article.children)[1]
    core_text = list(text.children)[7]

    # Each section is 3 child objects at a time, ['\n','header', 'info'...next 3 etc]
    section_objs = list(core_text.children)
    sections_dict = OrderedDict()
    for i in range(0,len(section_objs)-1,3):
        section_header = (list(section_objs[i+1]))[0]
        ccode = section_header['ccode']
        sectiontitle = section_header['sectiontitle']
        section_dict = OrderedDict()
        current_subsection_heading = ""
        current_entry_heading = ""
        entry_dict = None
        section_content = section_objs[i+2]
        for subsec in list(section_content.children):
            if isinstance(subsec, element.NavigableString):
                continue
            if isinstance(subsec, element.Tag):
                class_name = subsec.attrs.get('class')
                id_name = subsec.attrs.get('id')
                if class_name is None: # SPANS case
                    spans = list(subsec.children)
                    for sp in spans:
                        span_class_name = sp.attrs.get('class')
                        if span_class_name is None:
                            continue
                        if "category" in span_class_name:
                            current_entry_heading = depunctualize_text(sp.text)
                            if entry_dict is None:
                                entry_dict = OrderedDict()
                            entry_dict[current_entry_heading] = ""
                        if "category_data" in span_class_name and current_entry_heading != "":
                                entry_dict[current_entry_heading] = depunctualize_text(sp.text)
                    continue
                # DIV case
                # if "category" in class_name and "sas_light" in class_name:
                if "category" in class_name and "field" in id_name:
                    if(entry_dict != None) and (len(entry_dict.items()) > 0):
                        section_dict[current_subsection_heading] = entry_dict #store current entry dict
                    current_subsection_heading = depunctualize_text(subsec.text)
                    entry_dict = OrderedDict()
                    current_entry_heading = ""
                    # entry_dict[current_entry_heading] = ""
                if "category_data" in class_name and current_subsection_heading != "":
                    if entry_dict.get(current_entry_heading) == None:
                        entry_dict[current_entry_heading] = ""
                    entry_dict[current_entry_heading] += depunctualize_text(subsec.text)

        if (entry_dict != None) and len(entry_dict.values()) > 0:
            section_dict[current_subsection_heading] = entry_dict  # store current entry dict
        sections_dict[sectiontitle] = section_dict
    return sections_dict

def serialize_all_info(info_dict):
    json_info = json.dumps(all_info, indent=4)
    # Skipping middle level categories, add onlu low level as columns
    country_list = []
    for country, cdict in info_dict.items():
        new_dict = OrderedDict()
        for c, ldict in cdict.items():
            for col, valdict in ldict.items():
                val = " ; ".join(valdict.values())
                # print("{}: {} : {}".format(country,col,val))
                new_dict["Country"] = country
                new_dict[col] = val
        country_list.append(new_dict)
    df = pd.DataFrame(country_list)
    return json_info, df

def depunctualize_text(oldtext):
    s = oldtext
    s = s.replace(":","")
    s = s.replace(",",";")
    return s

if __name__ == "__main__":
    all_info = collect_country_wise_information()
    json_info, df = serialize_all_info(all_info)
    # print(json_info)
    # print(ccodes())
    print(df.head())
    df.to_csv("data/countries.csv")