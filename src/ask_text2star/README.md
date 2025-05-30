# RAG on Paired Data

A generic RAG (Retrieval Augmented Generation) framework to fetch responses from multiple formats such as docs, code, SQL and testcases.

<!-- 
## Prompt
```
You are an expert in llamaindex coding. Need to build chatbot application PoC using streamlit ui, open source models from groq and llamaindex.



Here is the problem for which code needs to be written using the RAG(retrieval augmented generation) approach. Here RAG training or indexing data is in MS Excel with two columns format. First column is English query and second column is of the response. The type of response can vary as below:

-  paragraph for FAQ chatbot

-  sql for text 2 sql application 

-  testing steps for QA application 

-  code blog for coding application 



Generate sample excel for all the above types, read them in python code, index them chormadb and build full end to end application using llamaindex framework



Here indexing is done for the first column primarily and when user query comes, it's matched in index for the first column and top k corresponding answers from second column are brought in the prompt for response generation. If there is any ready class for such Rag approach, use it or code it from scratch.
```

Refinement

```
Split the code in app.py into two files app.py which contains primarliy srtealit code and then rag.py for rag related llamaindex code, make rag code into a class, test it with some examples in __main__ below that file , streamlit from app.y should just instantiate rag class and call it's member functions
``` 
-->

**How the RAG works here?:**

1.  **User Query:** You type a question.
2.  **Embedding:** Your query is converted into a vector by the `HuggingFaceEmbedding` model.
3.  **Similarity Search:** This vector is compared against all the vectors of the "English Query" column entries stored in ChromaDB.
4.  **Retrieval:** The `top_k` most similar "English Query" entries are retrieved along with their associated `metadata` (which contains your "Response" from Excel).
5.  **Prompt Augmentation:** The retrieved queries and their responses are formatted into a prompt for the Groq LLM.
6.  **Generation:** The Groq LLM uses this augmented prompt to generate a final, coherent answer. It's guided to use the direct Excel responses if they are highly relevant.

-----

** What makes it different?:**
✅ One System, Multiple Use Cases: FAQ responses, SQL generation, QA testing steps, and code snippets 
✅ Excel-Powered Knowledge Base: Non-technical users can easily update training data 
✅ Modern Tech Stack: LlamaIndex + ChromaDB + Groq + Streamlit 
✅ Cost-Effective: Leverages open-source models with fast inference 
✅ Production-Ready?: Complete implementation is there but of academic level, not ready to use or is production worthy.

Read corresponding article: https://medium.com/analytics-vidhya/building-a-multi-format-rag-app-7dd70bc8daca


## Prerequisites:

- Install libraries:

```
pip install llama-index streamlit pandas openpyxl llama-index-embeddings-huggingface llama-index-llms-groq llama-index-vector-stores-chromadb chromadb
```
- Groq API Key:
	- Sign up at Groq Cloud.
	- Get an API key.
	- Set it as an environment variable: GROQ_API_KEY="YOUR_API_KEY"
	
	
Okay, I can help you build this RAG chatbot PoC\! 🧑‍💻

Here's how we'll approach it:

1.  **Sample Excel Data:** I'll define the structure and provide examples for your four types of Q\&A.
2.  **Core LlamaIndex & Groq Logic:** We'll set up the environment, load data, index it using ChromaDB, and configure a RAG pipeline with a Groq LLM.
3.  **Streamlit UI:** We'll wrap everything in a simple Streamlit interface.

Let's get started\!

<!-- 
-----

## 📝 Sample Excel Data

You'll need to create four separate Excel files. Here's the structure and some examples. Let's name them:

  * `faq_data.xlsx`
  * `text2sql_data.xlsx`
  * `qa_testing_data.xlsx`
  * `coding_blog_data.xlsx`

**Common Structure (for all files):**

Two columns:

  * **Column A:** `English Query`
  * **Column B:** `Response`

-----

**1. `faq_data.xlsx` (FAQ Chatbot)**

| English Query                                    | Response                                                                                                                                                              |
| :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What are your business hours?                    | Our business hours are Monday to Friday, 9 AM to 6 PM. We are closed on weekends and public holidays.                                                                  |
| How can I reset my password?                     | To reset your password, go to the login page and click on the "Forgot Password" link. Follow the instructions sent to your registered email address.                     |
| What is the return policy?                       | You can return most new, unopened items within 30 days of delivery for a full refund. We'll also pay the return shipping costs if the return is a result of our error. |
| Where are you located?                           | Our main office is located at 123 Tech Park, Silicon Valley, CA 94000.                                                                                                 |
| How do I contact customer support?               | You can contact customer support by emailing support@example.com or calling our toll-free number at 1-800-555-1234 during business hours.                               |
| What payment methods do you accept?              | We accept Visa, MasterCard, American Express, PayPal, and direct bank transfers.                                                                                      |
| How long does shipping take?                     | Standard shipping usually takes 3-5 business days within the continental US. International shipping times may vary.                                                     |
| Can I change my shipping address after ordering? | If your order has not yet shipped, you may be able to change the shipping address by contacting customer support immediately.                                             |
| Do you offer gift wrapping?                      | Yes, we offer gift wrapping services for a small additional fee. You can select this option during checkout.                                                            |
| What is your privacy policy?                     | You can find our detailed privacy policy on our website at [example.com/privacy](https://www.google.com/search?q=https://example.com/privacy). It outlines how we collect, use, and protect your personal information.                 |

-----

**2. `text2sql_data.xlsx` (Text to SQL Application)**

| English Query                                         | Response                                                                     |
| :---------------------------------------------------- | :--------------------------------------------------------------------------- |
| Show me all users from New York                       | `SELECT * FROM Users WHERE City = 'New York';`                               |
| How many orders were placed last month?               | `SELECT COUNT(OrderID) FROM Orders WHERE OrderDate >= date('now', '-1 month');` |
| List customers who spent more than $1000              | `SELECT CustomerName FROM Customers WHERE TotalSpent > 1000;`                  |
| Find products in the 'Electronics' category           | `SELECT ProductName FROM Products WHERE Category = 'Electronics';`            |
| What is the average price of laptops?                 | `SELECT AVG(Price) FROM Products WHERE ProductName LIKE '%Laptop%';`          |
| Show active employees hired before 2020               | `SELECT * FROM Employees WHERE Status = 'Active' AND HireDate < '2020-01-01';` |
| Count the number of products in stock                 | `SELECT SUM(StockQuantity) FROM Inventory;`                                  |
| List all suppliers from California                    | `SELECT SupplierName FROM Suppliers WHERE State = 'CA';`                     |
| Find the most recent order                            | `SELECT * FROM Orders ORDER BY OrderDate DESC LIMIT 1;`                      |
| Show me users whose names start with 'J' and live in 'Texas' | `SELECT * FROM Users WHERE FirstName LIKE 'J%' AND State = 'Texas';`      |

-----

**3. `qa_testing_data.xlsx` (QA Testing Steps Application)**

| English Query                                  | Response                                                                                                                                                                                                                                                                                                                                                        |
| :--------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How to test user login functionality?          | 1. Navigate to the login page. \\n2. Enter valid username and password. \\n3. Click 'Login'. \\n4. Verify successful login and redirection to dashboard. \\n5. Enter invalid username and valid password. \\n6. Click 'Login'. \\n7. Verify error message. \\n8. Enter valid username and invalid password. \\n9. Click 'Login'. \\n10. Verify error message.         |
| Steps to verify search product feature?        | 1. Go to the product search page. \\n2. Enter a known product name in the search bar. \\n3. Click 'Search'. \\n4. Verify that the correct product is displayed. \\n5. Enter a partial product name. \\n6. Verify relevant products are displayed. \\n7. Enter a non-existent product name. \\n8. Verify 'No results found' message.                                   |
| How to test the add to cart functionality?     | 1. Navigate to a product page. \\n2. Select product quantity and options (if any). \\n3. Click 'Add to Cart'. \\n4. Verify the cart icon updates with the correct item count. \\n5. Navigate to the cart page. \\n6. Verify the added product is listed with correct details and price. \\n7. Try adding the same product again. \\n8. Verify quantity updates in cart. |
| Test steps for password reset via email        | 1. Go to the 'Forgot Password' page. \\n2. Enter a registered email address. \\n3. Click 'Send Reset Link'. \\n4. Check the user's email inbox for a password reset email. \\n5. Click the reset link in the email. \\n6. Verify redirection to the password reset page. \\n7. Enter a new password and confirm it. \\n8. Click 'Reset Password'. \\n9. Try logging in with the new password. |
| Verify checkout process with valid payment     | 1. Add items to the cart. \\n2. Proceed to checkout. \\n3. Fill in valid shipping and billing information. \\n4. Select a valid payment method. \\n5. Enter valid payment details. \\n6. Click 'Place Order'. \\n7. Verify order confirmation page is displayed. \\n8. Verify order confirmation email is received.                                                     |
| Test cases for user registration               | 1. Navigate to the registration page. \\n2. Fill all required fields with valid data. \\n3. Submit the form. \\n4. Verify successful registration and account creation. \\n5. Try registering with an existing email. \\n6. Verify appropriate error message. \\n7. Try registering with missing required fields. \\n8. Verify specific error messages for each field. |
| How to test filter functionality on product list? | 1. Go to the product listing page. \\n2. Apply a price range filter. \\n3. Verify only products within that range are shown. \\n4. Apply a category filter. \\n5. Verify products match the selected category. \\n6. Combine multiple filters. \\n7. Verify results match all filter criteria. \\n8. Clear all filters. \\n9. Verify all products are shown again. |
| Steps to test user profile update              | 1. Log in to the application. \\n2. Navigate to the user profile page. \\n3. Click 'Edit Profile'. \\n4. Modify some profile information (e.g., name, phone number). \\n5. Click 'Save Changes'. \\n6. Verify the updated information is displayed. \\n7. Log out and log back in. \\n8. Verify the changes persist.                                                |
| How to verify proper error handling for forms? | 1. Identify a form in the application. \\n2. Submit the form with empty required fields. \\n3. Verify appropriate error messages are shown for each missing field. \\n4. Submit the form with invalid data types (e.g., text in a number field). \\n5. Verify type-specific error messages. \\n6. Submit the form with data exceeding maximum length. \\n7. Verify length validation messages. |
| Testing sorting functionality on a data table  | 1. Navigate to a page with a data table (e.g., orders list). \\n2. Click on a column header that supports sorting (e.g., 'Date'). \\n3. Verify data sorts in ascending order. \\n4. Click the same column header again. \\n5. Verify data sorts in descending order. \\n6. Test with different sortable columns.                                                  |

-----

**4. `coding_blog_data.xlsx` (Coding Application)**

| English Query                                      | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How to write a Python function to read a CSV?      | `python\nimport csv\n\ndef read_csv_file(filepath):\n    \"\"\"Reads a CSV file and returns its content as a list of lists.\"\"\"\n    data = []\n    try:\n        with open(filepath, mode='r', newline='') as file:\n            csv_reader = csv.reader(file)\n            for row in csv_reader:\n                data.append(row)\n    except FileNotFoundError:\n        print(f\"Error: The file {filepath} was not found.\")\n    except Exception as e:\n        print(f\"An error occurred: {e}\")\n    return data\n\n# Example usage:\n# csv_data = read_csv_file('your_file.csv')\n# if csv_data:\n#     for row in csv_data:\n#         print(row)\n`                                                                                                   |
| Javascript snippet for a simple API GET request?   | ``javascript\nasync function fetchData(url) {\n    try {\n        const response = await fetch(url);\n        if (!response.ok) {\n            throw new Error(`HTTP error! status: ${response.status}`);\n        }\n        const data = await response.json();\n        console.log(data);\n        return data;\n    } catch (error) {\n        console.error('Error fetching data:', error);\n    }\n}\n\n// Example usage:\n// fetchData('https://api.example.com/data');\n``                                                                                                                                                           |
| Basic Flask app structure in Python                | `python\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello_world():\n    return 'Hello, World!'\n\n@app.route('/about')\ndef about():\n    return 'This is a simple Flask application.'\n\nif __name__ == '__main__':\n    app.run(debug=True)\n`\\n\\nTo run this:\\n1. Save as `app.py`.\\n2. Open terminal in the same directory.\\n3. Run `python app.py`.\\n4. Open your browser and go to `http://127.0.0.1:5000/`.                                                                                                                                                           |
| How to create a React functional component?        | `javascript\nimport React from 'react';\n\nconst MyComponent = (props) => {\n    return (\n        <div>\n            <h1>Hello, {props.name}!</h1>\n            <p>This is a functional component.</p>\n        </div>\n    );\n};\n\nexport default MyComponent;\n\n// To use this component:\n// import MyComponent from './MyComponent';\n// <MyComponent name=\"User\" />\n`                                                                                                                                                                                                                                                                         |
| SQL query to select distinct values from a column  | `sql\nSELECT DISTINCT column_name\nFROM table_name;\n\n-- Example:\n-- To get all unique cities from a Customers table:\n-- SELECT DISTINCT City\n-- FROM Customers;\n`                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Python class for a simple linked list node         | `python\nclass Node:\n    def __init__(self, data=None):\n        self.data = data\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n\n    def append(self, data):\n        new_node = Node(data)\n        if not self.head:\n            self.head = new_node\n            return\n        last_node = self.head\n        while last_node.next:\n            last_node = last_node.next\n        last_node.next = new_node\n\n    def display(self):\n        elements = []\n        current_node = self.head\n        while current_node:\n            elements.append(current_node.data)\n            current_node = current_node.next\n        print(\" -> \".join(map(str, elements)))\n\n# Example:\n# ll = LinkedList()\n# ll.append(1)\n# ll.append(2)\n# ll.append(3)\n# ll.display() # Output: 1 -> 2 -> 3\n` |
| How to handle exceptions in Python using try-except | `python\ndef divide_numbers(a, b):\n    try:\n        result = a / b\n        print(f\"The result is: {result}\")\n    except ZeroDivisionError:\n        print(\"Error: Cannot divide by zero.\")\n    except TypeError:\n        print(\"Error: Invalid input types. Both inputs must be numbers.\")\n    except Exception as e:\n        print(f\"An unexpected error occurred: {e}\")\n    finally:\n        print(\"Division operation attempted.\")\n\n# Examples:\n# divide_numbers(10, 2)  # Output: The result is: 5.0\n# divide_numbers(10, 0)  # Output: Error: Cannot divide by zero.\n# divide_numbers(10, 'a') # Output: Error: Invalid input types...\n` |
| Generate a random number in Python                 | `python\nimport random\n\n# Generate a random integer between a and b (inclusive)\nrandom_integer = random.randint(1, 100)\nprint(f\"Random integer: {random_integer}\")\n\n# Generate a random floating point number between 0.0 and 1.0\nrandom_float = random.random()\nprint(f\"Random float (0-1): {random_float}\")\n\n# Generate a random float from a range\nrandom_float_range = random.uniform(5.0, 10.0)\nprint(f\"Random float (5.0-10.0): {random_float_range}\")\n\n# Choose a random element from a sequence\nmy_list = ['apple', 'banana', 'cherry']\nrandom_choice = random.choice(my_list)\nprint(f\"Random choice from list: {random_choice}\")\n` |
| How to read a JSON file in Python                | `python\nimport json\n\ndef read_json_file(filepath):\n    \"\"\"Reads a JSON file and returns its content.\"\"\"\n    data = None\n    try:\n        with open(filepath, 'r') as file:\n            data = json.load(file)\n    except FileNotFoundError:\n        print(f\"Error: The file {filepath} was not found.\")\n    except json.JSONDecodeError:\n        print(f\"Error: Could not decode JSON from the file {filepath}.\")\n    except Exception as e:\n        print(f\"An error occurred: {e}\")\n    return data\n\n# Example usage:\n# Assume 'data.json' contains: {\"name\": \"John\", \"age\": 30}\n# json_data = read_json_file('data.json')\n# if json_data:\n#     print(json_data['name']) # Output: John\n` |
| Simple HTML boilerplate                            | `html\n<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name-=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>My Web Page</title>\n    <link rel=\"stylesheet\" href=\"style.css\"> \n</head>\n<body>\n    <header>\n        <h1>Welcome to My Page</h1>\n    </header>\n    \n    <nav>\n        <ul>\n            <li><a href=\"#\">Home</a></li>\n            <li><a href=\"#\">About</a></li>\n            <li><a href=\"#\">Services</a></li>\n            <li><a href=\"#\">Contact</a></li>\n        </ul>\n    </nav>\n\n    <main>\n        <section>\n            <h2>Main Content</h2>\n            <p>This is a paragraph in the main content area.</p>\n        </section>\n    </main>\n\n    <footer>\n        <p>&copy; 2025 My Web Page. All rights reserved.</p>\n    </footer>\n\n    <script src=\"script.js\"></script> \n</body>\n</html>\n`\\nCreate `style.css` for CSS and `script.js` for JavaScript. |

-----

## 🐍 Python Code (LlamaIndex, Groq, Streamlit)

Now, let's write the Python script.

**Prerequisites:**

1.  **Install libraries:**
    ```bash
    pip install llama-index streamlit pandas openpyxl llama-index-embeddings-huggingface llama-index-llms-groq llama-index-vector-stores-chromadb chromadb
    ```
2.  **Groq API Key:**
      * Sign up at [Groq Cloud](https://console.groq.com/keys).
      * Get an API key.
      * Set it as an environment variable: `GROQ_API_KEY="YOUR_API_KEY"`

**Python Script (`app.py`):**

**Explanation and Key Changes:**

1.  **`dotenv` for API Key:** Added `python-dotenv` to manage the `GROQ_API_KEY` if you place it in a `.env` file in the same directory as `app.py`. Create a `.env` file with:
    ```
    GROQ_API_KEY="your_actual_groq_api_key"
    ```
2.  **LlamaIndex Settings (`configure_llamaindex`)**:
      * Uses `HuggingFaceEmbedding` with `sentence-transformers/all-MiniLM-L6-v2` for efficient local embeddings.
      * Uses `Groq` for the LLM, configured with your API key and chosen model (`llama3-8b-8192` is a good default).
      * Sets a global `Settings.chunk_size`.
      * The function is cached with `@st.cache_resource` so settings are initialized once.
3.  **Data Loading (`load_data_from_excel`)**:
      * Reads the "English Query" and "Response" columns.
      * Creates `llama_index.core.Document` objects.
      * **Crucially**:
          * `text=query`: The content of the document that gets embedded and searched against is the "English Query".
          * `metadata={"response": response, "filename": ...}`: The actual "Response" from your Excel sheet is stored in the document's metadata. This is key for our RAG approach. We also store the filename for traceability.
4.  **Vector Index Creation (`create_vector_index`)**:
      * Uses `ChromaVectorStore` for local, persistent storage.
      * A `db_path` and `collection_name` are specified to allow the index to be saved and reloaded across sessions.
      * It tries to `get_collection` first; if it fails (e.g., `DoesNotExistError` or similar, though `chromadb` might raise its own specific errors which a general `except` catches), it creates a new one.
      * `VectorStoreIndex.from_documents` is used to build the index if it's new.
      * `VectorStoreIndex.from_vector_store` is used if loading an existing one.
      * The function is cached with `@st.cache_resource`.
5.  **RAG Querying (`query_rag_index`)**:
      * It takes the user query and retrieves the `top_k` most similar documents (which are your "English Query" rows) using `index.as_retriever()`.
      * **Core RAG Logic**:
          * It extracts the `response` from the `metadata` of each retrieved node.
          * It then constructs a detailed prompt for the Groq LLM. This prompt includes:
              * The original user query.
              * The retrieved "English Query" (as `Retrieved Question`) and its corresponding "Response" from Excel (as `Retrieved Answer`).
          * The LLM's task is to synthesize an answer based on this context, prioritizing direct answers if a retrieved question closely matches the user's query. This way, for exact or very similar FAQ-like questions, you'll get the pre-defined Excel response. For more nuanced queries, the LLM can combine or rephrase.
6.  **Streamlit UI**:
      * **File Uploader:** Allows uploading multiple Excel files.
      * **Processing Button:** Triggers data loading and index creation/loading.
      * **Persistent Index:** The script now tries to save the ChromaDB index to a local directory (`./chroma_db_excel_poc`) and load it if it exists on subsequent runs, even if no files are uploaded initially. This avoids re-indexing every time if the data hasn't changed.
      * **Chat Interface:** Standard Streamlit chat elements.
      * **Retrieved Information Display:** An expander shows the `matched_query`, the `response` from Excel, the similarity `score`, and the `source` file for each retrieved document, giving transparency.
      * **State Management (`st.session_state`):** Used to store the `vector_index`, `messages` for the chat, and `data_loaded` status across reruns.
      * **Error Handling:** Basic checks for API keys, file formats, and index availability.
7.  **Directory for ChromaDB:** The script will create a directory (e.g., `chroma_db_excel_poc`) in the same location as `app.py` to store the vector database. Ensure your environment has write permissions there. 
-->


**To Run the Application:**

1.  Save the Python code above as `app.py`.
2.  Create the sample `.xlsx` files in the same directory (or provide paths to them).
3.  Make sure your `GROQ_API_KEY` is set as an environment variable or in a `.env` file.
4.  Open your terminal in that directory and run:
    ```bash
    streamlit run app.py
    ```
5.  Your browser should open with the Streamlit application.
6.  Upload your Excel files, click "Process Files & Build/Load Index", and then start chatting\!

** Sample Questions:**
- What are your business hours?
- Show me all users from New York
- How to test user login functionality?
- How to write a Python function to read a CSV? 

This setup provides a good foundation. You can expand it further with more sophisticated data cleaning, advanced LlamaIndex features, or a more complex UI as needed. Good luck\! 👍