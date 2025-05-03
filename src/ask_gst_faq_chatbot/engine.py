
##dict of response for each type of intent
intent_response_dict = {
    "intro": ["This is a GST FAQ bot. One stop-shop to all your GST related queries"],
    "greet":["Hey","Hello","Hi"],
    "goodbye":["Bye","It was nice talking to you","See you","ttyl"],
    "affirm":["Cool","I know you would like it"]

}

gstinfo_response_dict = {
    "GST": " Goods and Service Tax (GST) is a destination based tax on consumption of goods and services.",
    "benefits":"GST consumes more than a dozen taxes, thus making it hassle free and efficient.",
    "faq_link":'You can check all the answers here <a href="http://www.cbec.gov.in/resources//htdocs-cbec/deptt_offcr/faq-on-gst.pdf</a>'
}

gst_query_value_dict = {
    "12%":"Non-AC hotels, business class air ticket, frozen meat products, butter, cheese, ghee, dry fruits in packaged form, animal fat, sausage, fruit juices, namkeen and ketchup",
    "5%":"railways, air travel, branded paneer, frozen vegetables, coffee, tea, spices, kerosene, coal, medicines",
    "18%":"AC hotels that serve liquor, telecom services, IT services, flavored refined sugar, pasta, cornflakes, pastries and cakes",
    "28%":"5-star hotels, race club betting,wafers coated with chocolate, pan masala and aerated water",
    "exempt":"education, milk, butter milk, curd, natural honey, fresh fruits and vegetables, flour, besan"
}

def gst_info(entities):
    if entities == None:
        return "Could not find out specific information about this ..." +  gstinfo_response_dict["faq_link"]
    if len(entities) == 1:
        return gstinfo_response_dict[entities[0]]
    return "Sorry.." + gstinfo_response_dict["faq_link"]

def gst_query(entities):
    if entities == None:
        return "Could not query this ..." + gstinfo_response_dict["faq_link"]
    for ent in entities:
        qtype = ent["type"]
        qval = ent["entity"]
        if qtype == "gst-query-value":
            return gst_query_value_dict[qval]

        return gstinfo_response_dict[entities[0]]
    return "Sorry.." + gstinfo_response_dict["faq_link"]