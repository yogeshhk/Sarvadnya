# Ask Bharat (History of Ancient India)

(Based on [this repo](https://github.com/AIAnytime/Llama2-Medical-Chatbot/blob/main/README.md))

## Prerequisites

Before you can start using the ASK BHARAT Bot, make sure you have the following prerequisites installed on your system:

- Python 3.11 or higher
- Required Python packages (you can install them using pip):
    - langchain
    - chainlit
    - sentence-transformers
    - faiss
    - PyPDF2 (for PDF document loading)

## Installation

1. Clone this repository to your local machine.

    ```bash
    git clone <repo path>
    cd ask_bharat
    ```

2. Create a Python virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the required language models and data. Please refer to the Langchain documentation for specific instructions on how to download and set up the language model and vector store.

5. Set up the necessary paths and configurations in your project, including the `DB_FAISS_PATH` variable and other configurations as per your needs.

## Getting Started

To get started with the Ask Bharat Bot, you need to:

1. Set up your environment and install the required packages as described in the Installation section.

2. Configure your project by updating the `DB_FAISS_PATH` variable and any other custom configurations in the code.

3. Prepare the language model and data as per the Langchain documentation.

4. Start the bot by running the provided Python script or integrating it into your application.

## Usage

The Ask Bharat Bot can be used for answering medical-related queries. To use the bot, you can follow these steps:

1. Start the bot by running your application or using the provided Python script.

2. Send a medical-related query to the bot.

3. The bot will provide a response based on the information available in its database.

4. If sources are found, they will be provided alongside the answer.

5. The bot can be customized to return specific information based on the query and context provided.

## Containerizing the application

1. Download the model that you want to use and save that in models folder

2. Change the model path in the code (Line 50)

3. change directory to the directory where the Dockerfile is present and Run the following command to build the docker image

``` docker build -t bharat . ```

4. Run the following command to run the docker image

``` docker run -p 8000:8080 bharat ```

5. Open the browser and go to http://localhost:8000 to access the bot.

6. Type your query in the text field and press Enter to get a response from the bot.


## Contributing

Contributions to the Ask Bharat Bot are welcome! If you'd like to contribute to the project, please follow these steps:

1. Fork the repository to your own GitHub account.

2. Create a new branch for your feature or bug fix.

3. Make your changes and ensure that the code passes all tests.

4. Create a pull request to the main repository, explaining your changes and improvements.

5. Your pull request will be reviewed, and if approved, it will be merged into the main codebase.

## License

This project is licensed under the MIT License.

---

For more information on how to use, configure, and extend the Ask Bharat Bot, please refer to the Langchain documentation or contact the project maintainers.

Happy coding with Ask Bharat Bot! 🚀
