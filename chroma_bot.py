import discord
import functools
import json
import os
import openai
from dotenv import load_dotenv
import time
import chromadb
import asyncio
from chromadb.config import Settings
from chromadb.errors import NoDatapointsException

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
collection = None

# This example requires the 'message_content' intent.
intents = discord.Intents.all()
intents.message_content = True

client_discord = discord.Client(intents=intents)
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory = "memory"
))

##chromaDB functions##
def create_collection(chroma_client):
    """Create a collection with a name specified by the user."""
    while True:
        name = input("Enter a name for the collection: ")
        if name in chroma_client.list_collections():
            print(f"Collection {name} already exists.")
            continue
        else:
            break
    collection = chroma_client.create_collection(name=name)
    print(f"Collection {name} created.")
    return collection, name


def add_documents_from_folder(collection, folder_path):
    """Add all text files in the specified folder to the collection."""
    documents = []
    metadatas = []
    ids = []
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            document = f.read()
        documents.append(document)
        metadata = {"filename": filename}
        metadatas.append(metadata)
        ids.append(filename)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"{len(documents)} documents added to the collection.")


# persist the collection to disk
def load_or_create_collection(chroma_client, collection_name):
    collection = chroma_client.create_collection(name=collection_name, get_or_create=True)
    persist_directory = "memory"
    print(f"Collection {collection_name} loaded or created.")
    print(f"Persisting collection to {persist_directory}...")
    collection.persist(persist_directory=persist_directory)

    os.makedirs(os.path.join("memory", collection_name), exist_ok=True)
    
    return collection


def query_text(user_message):
    """Query the collection for the specified text."""
    print(f"Querying the collection for '{user_message}'...")
    n_results = 3
    max_distance = 1.5
    relevant_results = []
    
    while max_distance > 0.5:
        try:
            query_results = collection.query(query_texts=[user_message], n_results=n_results)
            relevant_results = [doc for doc, dist in zip(query_results['documents'], query_results['distances']) if dist[0] <= max_distance]
            if len(relevant_results) > 0:
                break
            max_distance -= 0.25
        except NoDatapointsException:
            print("No results found.")
            return []

    # Chunk the results and query them
    chunked_results = []
    for result in relevant_results:
        try:
            chunked_query_results = collection.query(query_texts=[result[0]], n_results=n_results)
            chunked_results.extend([doc for doc, dist in zip(chunked_query_results['documents'], chunked_query_results['distances']) if dist[0] <= max_distance])
        except NoDatapointsException:
            pass

    # Sort chunked results based on distance and take top 5
    sorted_chunked_results = sorted(chunked_results, key=lambda x: x[1])[:5]

    # Print the relevant results
    if sorted_chunked_results:
        for query_result in sorted_chunked_results:
            print(f"Relevant result: {query_result[0]}")
    else:
        print("No relevant results found.")

    return sorted_chunked_results


class HermesBot(discord.Client):
    def __init__(self, collection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection = collection

    @client_discord.event
    async def on_ready(self):
        print(f'We have logged in as {self.user}')

    @client_discord.event
    async def on_message(self, message):
        # Check if the bot is not mentioned or if the message is from another bot
        if self.user not in message.mentions or message.author.bot or (message.channel.id != 889887649001771129 and message.channel.id != 1099070370671243384):
            return

        user_name = message.author.name
        user_message = message.content

        # Remove the mention of Hermes from the user_message
        user_message = user_message.replace(f'<@!{self.user.id}>', '').strip()

        # Query the collection using the provided user_message
        results = query_text(user_message)
        print(f"Query results: {results}")
        if not results:
            hermes_input = user_message
        else:
            # Flatten the list of lists and remove duplicates
            document_texts = list(set([doc[0] for doc in results]))
            hermes_input = user_message + "\n\n" + "\n\n".join(document_texts)

        async with message.channel.typing():
            await asyncio.sleep(15)
        response = hermes_response(hermes_input)

        # Reply to the user who mentioned Hermes
        await message.reply(response)


##OPENAI functions##
def hermes_response(hermes_input):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'You are Hermes, a wise cracking inapproriate avatar.You curse, make inapproriate jokes, speak like a mix of a gen-Xer and a milenial. You use emojis, 😉and emoticons :-D. You are here to have fun and answer questions about 3D printing and anything else the community asks. You are not a used car salemen, represent our brand well, but don''t oversell it. You are SME. Use apporpriate paragraphs with seperations for readability. Please respond to the following message from our discord as if you were our avatar, Hermes the messenger god, who has come to support this community with Atlas 3DSS:'},
            {'role': 'user', 'content': hermes_input},
        ],
    )
    return response.choices[0].message["content"]

##CLI functions##
def collection_menu(collection):
    while True:
        print(f"\nCollection '{collection.name}' Menu")
        print("-------------")
        print("2. Add documents to the collection")
        print("3. Launch bot")
        print("4. Delete the collection")
        print("5. Return to main menu")
        choice = input("Enter your choice: ")

        if choice == "2":
            folder_path = input("Enter the path to the folder containing the text files: ")
            if not os.path.isdir(folder_path):
                print("Error: the specified path is not a directory.")
                continue
            add_documents_from_folder(collection, folder_path)
            print(f"There are {collection.count()} documents in the collection.")
        elif choice == "3":
            if collection is not None:
                bot = HermesBot(collection, intents=intents)
                bot.run(os.getenv('DISCORD_TOKEN'))
            else:
                print("Error: No collection selected. Please create or load a collection before launching the bot.")
        elif choice == "4":
            return_to_main = input("Are you sure you want to delete the collection? Type 'yes' to confirm: ")
            if return_to_main.lower() == "yes":
                return
            else:
                continue
        elif choice == "5":
            break
        else:
            print("Error: invalid choice.")


##main##
def main():
     chroma_client = chromadb.Client(chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory = "memory"
    ))
while True:
        print("\nChromaDB Menu\n-------------")
        print("1. Create a collection")
        print("2. Add documents from a folder to a collection")
        print("3. Load a collection")
        print("4. Delete a collection")
        print("5. launch the bot")
        print("6. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            name = input("Enter a name for the collection: ")
            if name in chroma_client.list_collections():
                collection = chroma_client.get_or_create_collection(name=name)
                print(f"Collection {name} loaded.")
            else:
                collection = chroma_client.create_collection(name=name)
                print(f"Collection {name} created.")
                print(f"There are {collection.name.count(name)} documents in the collection")

                os.makedirs(os.path.join("memory", name))
        elif choice == "2":
            if not "collection" in locals():
                print("Error: no collection selected.")
                continue
            folder_path = input("Enter the path to the folder containing the text files: ")
            if not os.path.isdir(folder_path):
                print("Error: the specified path is not a directory.")
                continue
            add_documents_from_folder(collection, folder_path)
        elif choice == "3":
            current_databases = chroma_client.list_collections()
            if len(current_databases) == 0:
                print("No collections found.")
                continue
            print("Current collections:")
            ##print just the names of the collections
            for collection in current_databases:
                print(collection)
            name = input("Enter the name of the collection to load: ")
            try:
                collection = chroma_client.load_collection(name=name)
                print(f"Collection {name} loaded.")
                print(f"There are {collection.name.count(name)} documents in the collection")
                collection_menu(collection)
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "4":
            name = input("Enter the name of the collection to delete: ")
            try:
                chroma_client.delete_collection(name=name)
                print(f"Collection {name} deleted.")
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == "5":
            if collection is not None:
                bot = HermesBot(collection, intents=intents)
                bot.run(os.getenv('DISCORD_TOKEN'))
            else:
                print("Error: No collection selected. Please create or load a collection before launching the bot.")

        elif choice == "6":
            break
        else:
            print("Error: invalid choice.")
if __name__ == "__main__":
    main()
